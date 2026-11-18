"""H2S-NN on the two large link-prediction tasks (ogbn-mag P–P, H-Pokec U–U; card 2 D4b).

One data path for every method: the task is the baselines' `LargeLinkBundle` (`src/baselines/bundles.py:
build_bundle`), loaded by the same function their loop calls (`src/baselines/large_lp_training.py: load_large_lp`):
H2GB's graph of the task's NC dataset, the task files and the **message graph** (held-out pairs removed in both
orientations, whatever their relation). Nothing here rebuilds the split, the negatives or the message graph; this
module only flattens that message graph into H2S-NN's node / fact layout (as `src/data/dataset.py: build_dataset`)
and moves the type-local pairs to global node ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from src.data.fact_layer import apply_fact_layer, build_fact_layer
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.large_lp import held_out_fact_mask
from src.data.structures import PreparedH2SData

if TYPE_CHECKING:
    from src.configs.config import TrainConfig


@dataclass
class LargeLinkH2SData:
    """The message graph in H2S-NN's layout and the task's pairs in its global node ids."""

    task: str
    graph: PreparedH2SData          # the message graph (fact layer applied); LinkScopes runs on it
    node_type: str                  # both endpoints' type
    dst_range: tuple[int, int]      # global ids of that type (HGB's uniform tail range)
    train_pos: torch.Tensor         # [2, P] global ids
    val_pos: torch.Tensor
    val_neg: torch.Tensor
    test_pos: torch.Tensor
    test_2hop: torch.Tensor
    test_random: torch.Tensor
    num_target_relations: int = 1

    def held_out(self) -> torch.Tensor:
        """Validation and test positives `[2, H]` (global ids): never a fact of the message graph."""
        return torch.cat([self.val_pos, self.test_pos], dim=1)


def prepare_large_link_data(lp: Any, fact_layer_mode: str) -> LargeLinkH2SData:
    """`lp`: the bundle's `LargeLPData`. Flattens its message graph as the node-classification builder does (node
    features stacked per type, facts flattened, fact layer) and checks that no held-out pair is a fact."""
    raw_x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(lp.graph)
    edge_index, edge_type, relation_names, relation_tuples = flatten_edges(lp.graph, node_slices)
    start, end = node_slices[lp.node_type]

    def to_global(pairs: torch.Tensor) -> torch.Tensor:
        return pairs.long() + start

    pairs = {name: to_global(getattr(lp, name)) for name in
             ("train_pos", "val_pos", "val_neg", "test_pos", "test_2hop", "test_random")}
    held_out = torch.cat([pairs["val_pos"], pairs["test_pos"]], dim=1)
    leaked = int(held_out_fact_mask(edge_index, held_out).sum())
    if leaked:  # card 2 D4b stop: a held-out positive would reach a scope
        raise RuntimeError(f"{lp.task}: {leaked} facts of the message graph are held-out pairs")
    layer = build_fact_layer(edge_index, edge_type, relation_names, raw_x.size(0), fact_layer_mode)
    edge_index, edge_type = apply_fact_layer(layer, edge_index, edge_type)
    n = raw_x.size(0)
    none = torch.zeros(n, dtype=torch.bool)
    graph = PreparedH2SData(
        x=raw_x, y=torch.zeros(n, dtype=torch.long), node_type=node_type, node_type_names=node_type_names,
        node_slices=node_slices, feature_dims=feature_dims, edge_index=edge_index, edge_type=edge_type,
        relation_names=layer.kept_names, relation_tuples=tuple(relation_tuples[r] for r in layer.kept),
        train_mask=none, val_mask=none, test_mask=none, target_node_type=lp.node_type,
        target_indices=torch.arange(start, end), target_y=torch.zeros(end - start, dtype=torch.long),
        target_train_mask=torch.zeros(end - start, dtype=torch.bool),
        target_val_mask=torch.zeros(end - start, dtype=torch.bool),
        target_test_mask=torch.zeros(end - start, dtype=torch.bool),
        num_classes=0, is_multilabel=False, face_groups=(), metrics=torch.zeros(0), fact_layer=layer)
    return LargeLinkH2SData(task=lp.task, graph=graph, node_type=lp.node_type, dst_range=(start, end), **pairs)


def load_bundle_data(cfg: "TrainConfig") -> Any:
    """The baselines' data of the task: `build_bundle` (a `LargeLinkBundle`), then `load_large_lp` on its data root,
    exactly as `fit_large_lp` does."""
    from src.baselines.bundles import build_bundle
    from src.baselines.large_lp_training import load_large_lp
    from src.baselines.protocol import LINK_PREDICTION

    bundle = build_bundle(LINK_PREDICTION, cfg.data.dataset_name, seed=cfg.seed,
                          overrides={"data": {"root": str(cfg.data.root)}})
    return load_large_lp(bundle.dataset, bundle.metadata["data_root"])


def build_large_link_dataset(cfg: "TrainConfig") -> LargeLinkH2SData:
    return prepare_large_link_data(load_bundle_data(cfg), cfg.data.fact_layer)


__all__ = ["LargeLinkH2SData", "build_large_link_dataset", "load_bundle_data", "prepare_large_link_data"]
