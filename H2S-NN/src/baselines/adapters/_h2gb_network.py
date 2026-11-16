"""H2GB's own networks on H2GB's `hgt` batches (part 2 E1: HAN as H2GB runs it, decision S5; H2G-former).

H2GB builds a network from its global graphgym cfg, read both at construction and in the forward (e.g.
`hetero_gnn_model.py:178-200` reads `cfg.gnn.layers_mp`, `cfg.gnn.use_linear`; `hetero_raw_encoder.py:126-131` reads
`cfg.train.sampler`). `configure_from_yaml` therefore sets the cfg exactly as H2GB's main does for that method and
dataset (`set_cfg`, `merge_from_file` of `configs/<dataset>/<dataset>-<Method>.yaml`, `assert_cfg`), then overrides
only the tuned keys; it must be the **last** cfg change before training (the sampler helpers of `minibatch.py` reset the cfg, so
loaders are built first).

Stated difference: on the binary datasets (IEEE-CIS-G, H-Pokec, PDNS) H2GB's `create_model` gives one logit and a
BCE loss (`graphgym/model_builder.py:22-23`); here the network gets `num_classes` outputs and the shared loop's
cross-entropy (one loss rule for every baseline).
"""

from __future__ import annotations

from typing import Any

import torch


def configure_from_yaml(method: str, dataset: str, data: Any, target_type: str, device: torch.device,
                        overrides: dict[str, Any]) -> Any:
    """H2GB's cfg for `method` on `dataset` (its YAML), `overrides` (dotted keys) applied, the input widths of
    `data` (types with `x`) in `cfg.share.dim_in`. Returns the cfg."""
    from src.baselines.minibatch import h2gb_config_file

    import H2GB  # noqa: F401  - registers H2GB's modules and config groups
    from H2GB.graphgym.config import assert_cfg, cfg, set_cfg  # type: ignore[import-not-found]

    set_cfg(cfg)
    cfg.merge_from_file(str(h2gb_config_file(method, dataset)))
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)
    assert_cfg(cfg)  # as H2GB's load_cfg after merging (graphgym/config.py:510-537): heads default to the task
    cfg.device = str(device)
    cfg.dataset.task_entity = target_type
    cfg.share["dim_in"] = {t: int(data[t].x.size(1)) for t in data.node_types if data[t].get("x") is not None}
    return cfg


class DatasetShim:
    """`dataset[0]` (and `.data`), what H2GB's network constructors read."""

    def __init__(self, data: Any, name: str) -> None:
        self.data = data
        self.name = name

    def __getitem__(self, index: int) -> Any:
        if index != 0:
            raise IndexError(index)
        return self.data

    def __len__(self) -> int:
        return 1


def build_hetero_gnn(data: Any, dataset: str, num_classes: int) -> torch.nn.Module:
    """H2GB's `HeteroGNNModel` (`H2GB/network/hetero_gnn_model.py:60`) under the current cfg."""
    from H2GB.network.hetero_gnn_model import HeteroGNNModel  # type: ignore[import-not-found]

    from H2GB.graphgym.config import cfg  # type: ignore[import-not-found]

    return HeteroGNNModel(cfg.share["dim_in"], num_classes, DatasetShim(data, dataset))


def seed_predictions(model: torch.nn.Module, batch: Any, target_type: str) -> torch.Tensor:
    """H2GB's head output for the seed targets (`head/hetero_node.py:29-38`: the first `batch_size` rows)."""
    out = model(batch)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out[target_type]
    return out[: int(batch[target_type].batch_size)]


__all__ = ["DatasetShim", "build_hetero_gnn", "configure_from_yaml", "seed_predictions"]
