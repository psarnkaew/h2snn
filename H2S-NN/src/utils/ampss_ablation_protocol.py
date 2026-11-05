from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.configs.config import AMPSS_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class AMPSSAblationSpec:
    index: int
    name: str
    source: str
    patches: Mapping[str, object]
    frozen_stage1: bool


AMPSS_ABLATIONS = (
    AMPSSAblationSpec(
        1,
        "no_faces",
        "auto",
        {"model.ablate_no_faces": True},
        False,
    ),
    AMPSSAblationSpec(
        2,
        "manual_schemas_uniform_weights",
        "manual",
        {},
        False,
    ),
    AMPSSAblationSpec(
        3,
        "auto_pool_uniform_weights",
        "auto",
        {"discovery.instance_weight_mode": "uniform"},
        True,
    ),
    AMPSSAblationSpec(
        4,
        "auto_pool_schema_agnostic_weights",
        "auto",
        {"discovery.instance_weight_mode": "schema_agnostic"},
        True,
    ),
    AMPSSAblationSpec(
        5,
        "full_schema_conditioned_weights",
        "auto",
        {"discovery.instance_weight_mode": "schema_conditioned"},
        True,
    ),
)


def _set_nested(payload: dict[str, Any], dotted_key: str, value: object) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def build_ablation_config(
    *,
    auto_base: Mapping[str, Any],
    manual_base: Mapping[str, Any],
    spec: AMPSSAblationSpec,
    training_seed: int,
    output_dir: str,
    exp_name_prefix: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(auto_base if spec.source == "auto" else manual_base)
    model = payload.setdefault("model", {})
    model["ablate_no_faces"] = False

    if spec.source == "auto":
        discovery = payload.setdefault("discovery", {})
        discovery["enabled"] = True
        discovery["mode"] = "auto"
        discovery["selector_strategy"] = "probe"
        discovery["protocol_version"] = AMPSS_PROTOCOL_VERSION
        discovery["instance_weight_mode"] = "schema_conditioned"
    else:
        payload["discovery"] = {
            "enabled": False,
            "mode": "manual",
            "protocol_version": AMPSS_PROTOCOL_VERSION,
            "instance_weight_mode": "uniform",
        }

    for key, value in spec.patches.items():
        _set_nested(payload, key, value)
    payload["seed"] = int(training_seed)
    payload["output_dir"] = output_dir
    payload["exp_name"] = (
        f"{exp_name_prefix}_{spec.index:02d}_{spec.name}_seed{training_seed}"
    )
    # Repeated seeds must not silently select a different graph split.
    payload.setdefault("data", {})["run_index"] = int(
        payload.get("data", {}).get("run_index", 0)
    )
    payload["data"]["num_runs"] = 1
    return payload


def assert_frozen_stage1_manifests(
    manifests: Iterable[Mapping[str, Any]],
) -> None:
    """Require variants 3--5 to share selector units, pool, and face materialization."""
    frozen = [
        item
        for item in manifests
        if int(item.get("ablation_index", -1)) in range(3, 6)
    ]
    if not frozen:
        raise ValueError("No Stage-II ablation manifests (indices 3--5) were provided.")
    fields = (
        "protocol_version",
        "selector_unit_hash",
        "selector_support_hash",
        "final_materialization_hash",
        "selected_identities",
    )
    reference = {field: frozen[0].get(field) for field in fields}
    for manifest in frozen:
        current = {field: manifest.get(field) for field in fields}
        if current != reference:
            raise ValueError(
                "Stage-II ablations do not share a frozen Stage-I manifest: "
                f"expected {reference}, got {current}."
            )
    if reference["protocol_version"] != AMPSS_PROTOCOL_VERSION:
        raise ValueError(
            "Stage-II ablations require protocol_version="
            f"{AMPSS_PROTOCOL_VERSION!r}."
        )
