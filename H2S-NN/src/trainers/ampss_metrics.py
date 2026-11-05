from __future__ import annotations

from collections.abc import Sequence

import torch

from src.configs.config import TrainConfig
from src.models.h2snn_model import H2SNNOutput


def build_ampss_metrics(
    output: H2SNNOutput,
    discovery: object,
    face_groups: Sequence[object],
    cfg: TrainConfig,
    *,
    schema_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Summarize effective Stage-II weights without affecting optimization."""
    num_schemas = int(discovery.num_schemas)
    means = output.instance_weight_means
    stds = output.instance_weight_stds
    nonempty = output.instance_weight_nonempty_mask
    metrics = {"operator_ramp": float(output.operator_ramp)}

    face_counts = [0 for _ in range(num_schemas)]
    incidence_counts = [0 for _ in range(num_schemas)]
    for group_index, face_group in enumerate(face_groups):
        schema_index = int(discovery.schema_ids[group_index].item())
        face_counts[schema_index] += int(face_group.face_node_index.size(0))
        incidence_counts[schema_index] += int(face_group.face_edge_index.numel())
        incidence_counts[schema_index] += int(face_group.closure_edge_index.size(0))

    if means is not None and stds is not None and nonempty is not None:
        if not (means.shape == stds.shape == nonempty.shape):
            raise ValueError("AMPSS weight statistics must have aligned shapes.")
        if int(means.numel()) != num_schemas:
            raise ValueError("AMPSS weight statistics must contain one value per schema.")
        active = nonempty & torch.tensor(
            [count > 0 for count in face_counts],
            dtype=torch.bool,
            device=nonempty.device,
        )
        if bool(active.any()):
            counts = means.new_tensor(face_counts)[active]
            active_means = means[active]
            active_stds = stds[active]
            total = counts.sum().clamp_min(1.0)
            global_mean = (counts * active_means).sum() / total
            global_second = (
                counts * (active_stds.square() + active_means.square())
            ).sum() / total
            global_std = (global_second - global_mean.square()).clamp_min(0.0).sqrt()
            metrics["instance_weight_mean"] = float(global_mean.detach().item())
            metrics["instance_weight_std"] = float(global_std.detach().item())

    incidence_mass = discovery.schema_incidence_mass.detach().cpu()
    for schema_index in range(num_schemas):
        name = (
            schema_names[schema_index]
            if schema_names is not None and schema_index < len(schema_names)
            else f"schema_{schema_index}"
        )
        prefix = f"discovery_schema/{name}"
        if (
            means is not None
            and stds is not None
            and nonempty is not None
            and bool(nonempty[schema_index])
        ):
            metrics[f"{prefix}/mean_weight"] = float(
                means.detach()[schema_index].item()
            )
            metrics[f"{prefix}/std_weight"] = float(
                stds.detach()[schema_index].item()
            )
        metrics[f"{prefix}/materialized_faces"] = float(face_counts[schema_index])
        metrics[f"{prefix}/materialized_incidences"] = float(
            incidence_counts[schema_index]
        )
        incidence_value = float(incidence_mass[schema_index].item())
        metrics[f"{prefix}/incidence_mass"] = incidence_value
        if cfg.task_type == "graph_classification":
            metrics[f"{prefix}/graph_fold_mean_incidence"] = incidence_value
    return metrics

