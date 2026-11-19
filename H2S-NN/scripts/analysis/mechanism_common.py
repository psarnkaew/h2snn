"""Card 7 V5 / V6 (ablation plan v2, "Face effect", "Two-regime analysis"): shared evaluation-only helpers of
`face_effect.py` and `two_regime.py`. Nothing here trains or changes a model; the card 5 hooks
(`src/analysis/mechanism_hooks.py`) are used as they are.

- **Runs:** a run directory (`config.yaml` + `checkpoints/best.pt`, written by `scripts/run_h2snn_runs.py` and the
  main table's final seeds) is loaded with `src.analysis.model_loading.load_trained_model`, which rebuilds the run's
  graph (its `data.root`: the original data or a perturbed graph) and evaluation draw (its seed). The row of the
  ablation table is read from the configuration (`arm_of`, the changes of `scripts/ablation/arms.py`).
- **Evaluation views:** one evaluation forward over the targets read. Full batch (HGB): the whole evaluation graph,
  one view. Mini-batch (`data.training_mode: minibatch`; Freebase NC if card 5 D1 decides so): the trainer's own
  evaluation draw and scope batches (`MiniBatchNodeTrainer.evaluation_cells` / `_batch`), one view per batch of test
  targets. The training targets (R3's probe) use one draw with the run seed in the mini-batch case (the evaluation
  draw covers validation and test targets only). Every forward starts cold (`_warm_start` cleared), so a result does
  not depend on the forwards before it.
- **eta_v** (plan v2): the share of v's target-type neighbours through the M_2 meta-paths that have a different label;
  IMDB: the mean over genres of the binary disagreement. Neighbours: the distinct labelled u != v with (u, v) in the
  union of the meta-path graphs G_P over M_2, built by the reference code (`compute_heterophily.py`:
  `h2gb_metapaths`, `iter_metapath_chunks`, `labeled_mask`, `genre_labels`; Freebase on its symmetrised graph,
  `symmetrise_like_h2gb`, spec §1) on the run's own graph. Nodes without such a neighbour get NaN (excluded, counted).
- **Quartiles** of eta_v over the test targets of a graph: bins (-inf, q25], (q25, q50], (q50, q75], (q75, inf)
  (`numpy.quantile`, linear); ties can leave a bin small or empty, the counts are reported.
- **Correct:** single label: argmax = label; IMDB: the predicted genre set (logit > 0) equals the true set.
- **Centroids and margins** (F2): class centroids of the test x* of one model; margin_v = distance to the nearest
  other-class centroid - distance to the own-class centroid (Euclidean). Single-label datasets only.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for sub in ("ablation", "heterophily"):
    if str(PROJECT_ROOT / "scripts" / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / sub))

ABLATION_DATASETS: tuple[str, ...] = ("acm", "dblp", "imdb", "freebase")
SINGLE_LABEL: tuple[str, ...] = ("acm", "dblp", "freebase")
from src.configs.config import EVAL_BATCH_TARGETS  # noqa: E402  (the trainer's evaluation batch)
CHUNK_SIZE = 200                    # calcHomophily's default (compute_heterophily `--chunk-size`)
QUARTILES: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")


# ------------------------------------------------------------------------------------------------ runs
def arm_of(cfg) -> str:
    """The ablation row of a run configuration (the one change of `arms.ARM_CHANGES`)."""
    model_arm = str(cfg.model.arm)
    identity = bool(cfg.ablation.fact_maps_identity)
    if model_arm == "no_path_cells":
        return "S0" if identity else "no_path_cells"
    if model_arm == "instance_features":
        return "instance_features"
    if identity:
        return "S2"
    if str(cfg.ablation.path_weight_input) == "endpoints":
        return "endpoints"
    if str(cfg.data.path_source) == "schemas":
        return "literature"
    if model_arm != "full":
        raise ValueError(f"out-of-scope arm {model_arm!r}")
    return "full"


@dataclass
class Run:
    run_dir: Path
    model: Any
    data: Any                       # the whole evaluation graph (PreparedH2SData)
    cfg: Any
    dataset: str
    seed: int
    arm: str
    data_root: Path

    @property
    def minibatch(self) -> bool:
        return self.cfg.data.training_mode == "minibatch"

    @property
    def multilabel(self) -> bool:
        return bool(self.data.is_multilabel)

    @property
    def graph_tag(self) -> str:
        """`original`, or `<kind>_h<level>` for a perturbed graph (`<...>/<kind>_h<level>/data`)."""
        return graph_tag_of(self.data_root)

    def graph_record(self) -> dict[str, Any] | None:
        import json

        path = self.data_root.resolve().parent / "graph.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def graph_tag_of(data_root: Path) -> str:
    """`<kind>_h<level>` / `original` for a folder with `graph.json` (`<...>/<folder>/data`), else `original`."""
    parent = Path(data_root).resolve().parent
    return parent.name if (parent / "graph.json").exists() else "original"


def run_info(run_dir: Path) -> dict[str, Any]:
    """dataset, seed, arm, data root and graph tag of a run, from its `config.yaml` alone (no model is built)."""
    import yaml

    from src.configs.config import TrainConfig

    cfg = TrainConfig.from_dict(yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8")))
    return {"run_dir": Path(run_dir), "dataset": str(cfg.data.dataset_name).lower(), "seed": int(cfg.seed),
            "arm": arm_of(cfg), "data_root": Path(cfg.data.root), "graph_tag": graph_tag_of(Path(cfg.data.root)),
            "task_type": cfg.task_type}


def load_run(run_dir: Path, device: str = "cpu", tag: str = "best") -> Run:
    from src.analysis.model_loading import load_trained_model

    model, data, cfg = load_trained_model(Path(run_dir), device=device, tag=tag)
    if cfg.task_type != "node_classification":
        raise ValueError(f"{run_dir}: node classification only, got {cfg.task_type!r}")
    model.eval()
    return Run(Path(run_dir), model, data, cfg, str(cfg.data.dataset_name).lower(), int(cfg.seed), arm_of(cfg),
               Path(cfg.data.root))


# ------------------------------------------------------------------------------------------------ views
@dataclass
class View:
    """One evaluation forward: `data` (the whole graph or a batch of scopes), the targets read in it (`positions`:
    rows of the forward's logits, `rows`: their node rows in `data`) with their global ids and labels."""
    data: Any
    positions: torch.Tensor
    rows: torch.Tensor
    ids: torch.Tensor
    labels: torch.Tensor


def split_mask(data, split: str) -> torch.Tensor:
    return {"train": data.target_train_mask, "val": data.target_val_mask, "test": data.target_test_mask}[split]


def _trainer(run: Run):
    from src.trainers.minibatch_trainer import MiniBatchNodeTrainer

    return MiniBatchNodeTrainer(run.model, None, run.cfg, None, None)


def views(run: Run, split: str = "test", targets: torch.Tensor | None = None,
          batch_size: int = EVAL_BATCH_TARGETS) -> Iterator[View]:
    """The evaluation views of `split` (or of the given global target ids, in that order)."""
    data = run.data
    if targets is None:
        targets = data.target_indices[split_mask(data, split)]
    targets = torch.as_tensor(targets, dtype=torch.long).cpu()
    position_of = {int(v): i for i, v in enumerate(data.target_indices.tolist())}
    if not run.minibatch:
        positions = torch.tensor([position_of[int(v)] for v in targets], dtype=torch.long)
        yield View(data, positions, targets, targets, data.target_y.cpu()[positions])
        return
    tr = _trainer(run)
    if split == "train":
        cells = tr._cells(data, targets, run.cfg.seed)          # one draw with the run seed (module docstring)
    else:
        cells = tr.evaluation_cells(data)
    for start in range(0, targets.numel(), batch_size):
        part = targets[start:start + batch_size]
        _, batch = tr._batch(data, part, cells)
        k = part.numel()
        yield View(batch, torch.arange(k), batch.target_indices.cpu(), part, batch.target_y.cpu())


def cold_forward(model, data):
    """One evaluation forward without the solver's warm start (the result does not depend on earlier forwards)."""
    if model.training:
        raise ValueError("evaluation forwards only")
    model._warm_start = None
    with torch.no_grad():
        return model(data, epoch=-1)


def read(model, view: View) -> dict[str, torch.Tensor]:
    """logits, x*, h0 of the view's targets (CPU, float64)."""
    out = cold_forward(model, view.data)
    rows = view.rows.to(out.x_star.device)
    return {"ids": view.ids, "labels": view.labels,
            "logits": out.logits[view.positions.to(out.logits.device)].double().cpu(),
            "x_star": out.x_star[rows].double().cpu(), "h0": out.h0[rows].double().cpu()}


def evaluate(run: Run, split: str = "test", targets: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    parts = [read(run.model, v) for v in views(run, split, targets)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def correct(logits: torch.Tensor, labels: torch.Tensor, multilabel: bool) -> np.ndarray:
    if multilabel:
        return ((logits > 0) == (labels > 0)).all(dim=1).cpu().numpy()
    return (logits.argmax(dim=1) == labels).cpu().numpy()


def labelled_targets(data) -> dict[int, Any]:
    """global id -> label (an int, or a tuple of genres) of every labelled target (train, validation or test)."""
    mask = data.target_train_mask | data.target_val_mask | data.target_test_mask
    ids = data.target_indices[mask].tolist()
    ys = data.target_y[mask]
    if ys.dim() > 1:
        return {v: tuple(int(a) for a in y.tolist()) for v, y in zip(ids, ys)}
    return {v: int(y) for v, y in zip(ids, ys.tolist())}


# ------------------------------------------------------------------------------------------------ eta_v
def reference_graph(data_root: Path, dataset: str):
    """(PyG graph as the heterophily reference loads it, target type, reference module); Freebase symmetrised."""
    import compute_heterophily as ref

    data = ref.load_hgb(Path(data_root), dataset)
    if dataset == "freebase":
        data = ref.symmetrise_like_h2gb(data)
    return data, ref.target_type(data), ref


def metapath_neighbour_pairs(data, target: str, ref, chunk_size: int = CHUNK_SIZE) -> Iterator[torch.Tensor]:
    """Distinct (u, v) pairs of the union of G_P over M_2 (target-local ids, u != v), one row range at a time (the
    chunks of `iter_metapath_chunks` share their row ranges across meta-paths: every path starts at the target)."""
    paths = ref.h2gb_metapaths(data, target)
    n = int(data[target].num_nodes)
    for chunks in zip(*(ref.iter_metapath_chunks(data, p, chunk_size) for p in paths)):
        pairs = torch.cat(chunks, dim=1)
        pairs = pairs[:, pairs[0] != pairs[1]]
        key = torch.unique(pairs[0] * n + pairs[1])
        yield torch.stack([key // n, key % n])


def eta(data_root: Path, dataset: str) -> dict[str, np.ndarray]:
    """eta_v per target (target-local index) and the number of labelled neighbours; NaN without one."""
    data, target, ref = reference_graph(data_root, dataset)
    y = data[target].y
    labelled = ref.labeled_mask(y)
    n = int(data[target].num_nodes)
    disagree = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.float64)
    multi = y.dim() > 1
    yb = (y > 0) if multi else y
    for u, v in metapath_neighbour_pairs(data, target, ref):
        keep = labelled[u] & labelled[v]
        u, v = u[keep], v[keep]
        if multi:
            d = (yb[u] != yb[v]).double().mean(dim=1)              # mean over genres of the binary disagreement
        else:
            d = (yb[u] != yb[v]).double()
        np.add.at(disagree, v.numpy(), d.numpy())
        np.add.at(count, v.numpy(), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(count > 0, disagree / np.maximum(count, 1.0), np.nan)
    return {"eta": values, "neighbours": count, "target": target}


def eta_of_ids(eta_values: np.ndarray, data, ids: torch.Tensor) -> np.ndarray:
    """eta_v of global node ids (PreparedH2SData numbers each type's nodes in the reference loader's order)."""
    start, _ = data.node_slices[data.target_node_type]
    return eta_values[np.asarray(ids, dtype=np.int64) - int(start)]


def quartile_edges(values: np.ndarray) -> list[float]:
    defined = values[~np.isnan(values)]
    if defined.size == 0:
        return [math.nan] * 3
    return [float(q) for q in np.quantile(defined, [0.25, 0.5, 0.75])]


def quartile_of(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """0-3 per node (Q1 = lowest eta), -1 for NaN (no labelled meta-path neighbour)."""
    out = np.searchsorted(np.asarray(edges), values, side="left").astype(np.int64)
    out[np.isnan(values)] = -1
    return out


# ------------------------------------------------------------------------------------------------ margins
def centroids(x: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[C, d] class means of `x` (NaN rows for a class with no node)."""
    out = torch.full((num_classes, x.size(1)), float("nan"), dtype=x.dtype)
    for c in range(num_classes):
        mask = labels == c
        if bool(mask.any()):
            out[c] = x[mask].mean(dim=0)
    return out


def margins(x: torch.Tensor, labels: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """distance to the nearest other-class centroid - distance to the own-class centroid."""
    dist = torch.cdist(x, centres.nan_to_num(nan=float("inf")))
    own = dist.gather(1, labels.view(-1, 1)).squeeze(1)
    other = dist.scatter(1, labels.view(-1, 1), float("inf")).min(dim=1).values
    return other - own


# ------------------------------------------------------------------------------------------------ summaries
def mean_range(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}


def auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """AUROC of `scores` separating `positive` from the rest; None with one group only."""
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(positive, scores))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)
