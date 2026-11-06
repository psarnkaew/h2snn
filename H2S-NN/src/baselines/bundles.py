"""The single place baseline data is loaded.

Every baseline sees identical splits, identical HGB two-hop negatives, and identical
folds, because all of them come through here and nothing re-loads a vendored copy of a
dataset. That is what makes the comparison tables defensible; if each adapter loaded its
own ACM, a difference in the table could be a difference in the data.

Bundles wrap the existing loaders rather than reimplementing them:

* node classification -> :func:`src.data.dataset.build_dataset`
* link prediction     -> :func:`src.data.link_prediction_dataset.build_link_prediction_dataset`
* graph classification-> :func:`src.data.graph_classification_dataset.build_graph_classification_dataset`

:class:`HomogeneousView` is the HGB-protocol projection used by the homogeneous baselines
(GCN, GAT, ChebNetII, NSD, CSNN). See its docstring for why it does not simply reuse
``src/data/hetero_flatten.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
)
from src.configs.config import TrainConfig

# HGB's --feats-type, from HGB/NC/benchmark/methods/GNN/run.py:
#   0 - loaded features
#   1 - only target node features (zero vec for others)
#   2 - only target node features (id vec for others)
#   3 - all id vec
FEATS_TYPE_CHOICES = (0, 1, 2, 3)

# Defaults taken from HGB/NC/benchmark/methods/GNN/README.md. That README is explicitly
# marked "(To be tuned)", so these are starting points; feats_type is also exposed in the
# homogeneous search space and selected on validation.
HGB_DEFAULT_FEATS_TYPE: dict[str, int] = {
    "acm": 2,
    "dblp": 3,
    "imdb": 0,
}

# Several baselines (PSHGCN, H2SGNN, SeHGNN, the HGB DGL models) read the raw HGB
# node.dat/link.dat/label.dat files directly through their own copy of HGB's data_loader
# rather than through PyG. Those files are already on disk as the PyG HGBDataset raw
# payload, so the adapters point at them instead of shipping a second copy.
HGB_RAW_DIRNAME: dict[str, str] = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB"}


def hgb_raw_dir(dataset: str) -> "Path":
    """Absolute path to one dataset's raw HGB directory."""

    from src.baselines.runtime import PROJECT_ROOT

    key = dataset.lower()
    if key not in HGB_RAW_DIRNAME:
        raise KeyError(
            f"No raw HGB directory known for {dataset!r}; expected one of "
            f"{sorted(HGB_RAW_DIRNAME)}."
        )
    name = HGB_RAW_DIRNAME[key]
    path = PROJECT_ROOT / "data" / key / key / "raw" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Raw HGB directory for {dataset!r} is missing: {path}")
    return path


@dataclass(slots=True)
class HomogeneousView:
    """A heterogeneous graph reduced the way HGB reduces it for homogeneous baselines.

    Two things here deliberately differ from ``src/data/hetero_flatten.py``, which exists
    to feed H2SNN's typed encoder rather than a type-agnostic model:

    1. **Per-type native-dimension features, not a padded matrix.**
       ``stack_node_features`` right-zero-pads every type to ``max_dim`` and concatenates.
       H2SNN gets away with that only because ``TypeSpecificNodeEncoder.forward`` slices
       ``x[mask][:, :type_dim]`` back out per type. HGB instead keeps each type at its own
       width and gives it its own ``nn.Linear(in_dim, hidden)``
       (HGB/NC/benchmark/methods/GNN/GNN.py). Handing a homogeneous model the padded
       matrix would make DBLP's 334-dim author features 8% of a 4231-dim input with the
       rest structural zeros, and the resulting number would measure our adapter.

    2. **Symmetrized and self-looped adjacency.**
       HGB builds ``dgl.DGLGraph(adjM + adjM.T)``, removes self-loops, then adds them,
       discarding edge direction and relation type. ``flatten_edges`` keeps both.

    ``features_list`` and ``in_dims`` are positionally aligned with ``node_type_names``.
    """

    features_list: list[torch.Tensor]
    in_dims: list[int]
    node_type_names: tuple[str, ...]
    node_slices: dict[str, tuple[int, int]]
    node_type: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    feats_type: int
    target_node_type: str

    @property
    def target_type_index(self) -> int:
        return self.node_type_names.index(self.target_node_type)

    @property
    def edge_index_without_self_loops(self) -> torch.Tensor:
        """The same graph with HGB's added self-loops removed.

        HGB adds self-loops because DGL's GraphConv needs them, and GCN/GAT/ChebNetII
        consume ``edge_index`` directly. Sheaf models build their own Laplacian and expect
        each undirected edge as an exact symmetric pair with no diagonal -- NSD asserts
        ``len(left_index) == edge_index.size(1) // 2``
        (neural-sheaf-diffusion/lib/laplace.py:343), which a self-loop violates.

        Same underlying graph either way; only the diagonal differs.
        """

        mask = self.edge_index[0] != self.edge_index[1]
        return self.edge_index[:, mask]


def _symmetrize_with_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Reproduce HGB's ``adjM + adjM.T`` -> remove self-loop -> add self-loop.

    Coalesced so a relation present in both directions does not become a doubled weight,
    matching HGB's construction from a summed sparse matrix.
    """

    both = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    both = both[:, both[0] != both[1]]
    unique = torch.unique(both, dim=1)
    loops = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    return torch.cat([unique, loops.expand(2, -1)], dim=1)


def build_homogeneous_view(
    *,
    x: torch.Tensor,
    node_type: torch.Tensor,
    node_type_names: tuple[str, ...],
    node_slices: dict[str, tuple[int, int]],
    feature_dims: dict[str, int],
    edge_index: torch.Tensor,
    target_node_type: str,
    feats_type: int,
) -> HomogeneousView:
    """Project a flattened heterogeneous graph onto the HGB homogeneous protocol."""

    if feats_type not in FEATS_TYPE_CHOICES:
        raise ValueError(
            f"feats_type must be one of {FEATS_TYPE_CHOICES}, got {feats_type!r}."
        )
    if target_node_type not in node_type_names:
        raise ValueError(
            f"target_node_type {target_node_type!r} is not among {node_type_names}."
        )

    num_nodes = int(x.size(0))
    target_index = node_type_names.index(target_node_type)

    features_list: list[torch.Tensor] = []
    in_dims: list[int] = []
    for type_index, type_name in enumerate(node_type_names):
        start, end = node_slices[type_name]
        count = end - start
        native_dim = int(feature_dims[type_name])
        keep_own = feats_type == 0 or (
            feats_type in {1, 2} and type_index == target_index
        )

        if keep_own and native_dim > 0:
            # Padding in stack_node_features is on the right, so the real attributes are
            # always the prefix columns [0, native_dim).
            features_list.append(x[start:end, :native_dim].clone())
            in_dims.append(native_dim)
            continue

        if feats_type == 1:
            # HGB uses a fixed 10-dim zero block for the non-target types.
            features_list.append(torch.zeros((count, 10), dtype=torch.float32))
            in_dims.append(10)
            continue

        # feats_type 2/3, and any type that has no attributes under feats_type 0:
        # a per-node identity vector. Kept sparse; densifying an n_t x n_t identity for
        # DBLP's 14328 papers would be a 1.6 GB allocation for no information.
        indices = torch.arange(count, dtype=torch.long)
        features_list.append(
            torch.sparse_coo_tensor(
                torch.stack([indices, indices]),
                torch.ones(count, dtype=torch.float32),
                (count, count),
            ).coalesce()
        )
        in_dims.append(count)

    return HomogeneousView(
        features_list=features_list,
        in_dims=in_dims,
        node_type_names=node_type_names,
        node_slices=dict(node_slices),
        node_type=node_type,
        edge_index=_symmetrize_with_self_loops(edge_index, num_nodes),
        num_nodes=num_nodes,
        feats_type=feats_type,
        target_node_type=target_node_type,
    )


@dataclass(slots=True)
class NodeBundle:
    """Node classification on one HGB dataset, with the project's canonical split."""

    dataset: str
    data: Any                                   # PreparedH2SData
    num_classes: int
    is_multilabel: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return NODE_CLASSIFICATION

    def homogeneous(self, feats_type: int | None = None) -> HomogeneousView:
        resolved = (
            feats_type
            if feats_type is not None
            else HGB_DEFAULT_FEATS_TYPE.get(self.dataset, 3)
        )
        return build_homogeneous_view(
            x=self.data.x,
            node_type=self.data.node_type,
            node_type_names=self.data.node_type_names,
            node_slices=dict(self.data.node_slices),
            feature_dims=dict(self.data.feature_dims),
            edge_index=self.data.edge_index,
            target_node_type=self.data.target_node_type,
            feats_type=resolved,
        )


@dataclass(slots=True)
class LinkBundle:
    """Link prediction on one HGB dataset.

    ``data.edge_index`` is already the training-positives-only message-passing graph, so
    an adapter that builds its structure from it cannot see validation or test positives.
    Adapters that derive their own structure (LatGRL's meta-path adjacencies) must derive
    it from this field for the same reason.
    """

    dataset: str
    data: Any                                   # PreparedLinkPredData
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return LINK_PREDICTION

    def homogeneous(self, feats_type: int | None = None) -> HomogeneousView:
        source_type = self.data.target_relation[0]
        return build_homogeneous_view(
            x=self.data.x,
            node_type=self.data.node_type,
            node_type_names=self.data.node_type_names,
            node_slices=dict(self.data.node_slices),
            feature_dims=dict(self.data.feature_dims),
            edge_index=self.data.edge_index,
            target_node_type=source_type,
            # feats_type 0 keeps whatever attributes a type actually has and falls back
            # to an identity encoding for types that have none. That is the right default
            # here: PubMed (200-dim) and Amazon (1156-dim) carry real features that a
            # blanket identity encoding would discard, while LastFM and YouTube are
            # attribute-free and collapse to identity under this branch anyway.
            feats_type=0 if feats_type is None else feats_type,
        )


@dataclass(slots=True)
class GraphBundle:
    """Graph classification on one TU dataset, for a single run index of the split.

    ``fold`` carries the PyG ``DataLoader`` triple already batched under the project's
    stratified 80/10/10 protocol, so a baseline inherits the exact splits H2SNN uses.
    """

    dataset: str
    fold: Any                                   # GraphClassificationFold
    num_classes: int
    run_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return GRAPH_CLASSIFICATION

    @property
    def in_channels(self) -> int:
        return int(self.fold.dims.in_channels)

    @property
    def num_node_types(self) -> int:
        return int(self.fold.dims.num_node_types)

    @property
    def loaders(self) -> tuple[Any, Any, Any]:
        return self.fold.train_loader, self.fold.val_loader, self.fold.test_loader


def _base_config(
    dataset: str,
    *,
    run_index: int,
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Build the canonical TrainConfig for one cell."""

    payload: dict[str, Any] = {
        "data": {"dataset_name": dataset, "run_index": run_index},
        "seed": seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if overrides:
        for key, value in overrides.items():
            if key == "data" and isinstance(value, dict):
                payload["data"].update(value)
            else:
                payload[key] = value
    return TrainConfig.from_dict(payload)


def build_bundle(
    task: str,
    dataset: str,
    *,
    run_index: int = 0,
    seed: int = 42,
    overrides: dict[str, Any] | None = None,
) -> NodeBundle | LinkBundle | GraphBundle:
    """Build the canonical data bundle for one (task, dataset) cell."""

    config = _base_config(dataset, run_index=run_index, seed=seed, overrides=overrides)

    if task == NODE_CLASSIFICATION:
        from src.data.dataset import build_dataset

        data = build_dataset(config)
        return NodeBundle(
            dataset=dataset,
            data=data,
            num_classes=data.num_classes,
            is_multilabel=data.is_multilabel,
            metadata={"split_seed": config.data.acm_split_seed},
        )

    if task == LINK_PREDICTION:
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        data = build_link_prediction_dataset(config)
        return LinkBundle(
            dataset=dataset,
            data=data,
            metadata={
                "target_relation": list(config.data.target_relation),
                "neg_sampling_mode": config.data.neg_sampling_mode,
            },
        )

    if task == GRAPH_CLASSIFICATION:
        from src.data.graph_classification_dataset import (
            build_graph_classification_dataset,
        )

        fold = build_graph_classification_dataset(config)
        return GraphBundle(
            dataset=dataset,
            fold=fold,
            num_classes=int(fold.dims.num_classes),
            run_index=run_index,
            metadata={
                "split_protocol": config.data.split_protocol,
                "split_seed": config.data.split_seed,
            },
        )

    raise ValueError(f"Unknown task {task!r}.")
