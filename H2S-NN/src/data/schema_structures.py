from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(slots=True)
class SchemaCandidate:
    schema_id: int
    name: str
    node_types: tuple[str, ...]
    relation_ids: tuple[int, ...]
    relation_names: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.relation_ids)


@dataclass(slots=True)
class SchemaStats:
    schema_id: int
    num_instances: int
    start_coverage: float
    end_coverage: float
    relation_diversity: float
    type_diversity: float
    avg_branching: float
    estimated_cost: float
    endpoint_label_disc: float       # mean ||y_hat_{v0} - y_hat_{vl}||^2; key topology signal
    start_node_sample: frozenset[int]  # retained for post-selection Jaccard diagnostics
    end_node_sample: frozenset[int]


@dataclass(slots=True)
class DiscoveryMetadata:
    """Sparse selected-schema support consumed by AMPSS Stage-II weighting."""

    num_schemas: int
    schema_ids: torch.Tensor
    schema_incidence_mass: torch.Tensor
    coverage_schema_index: torch.Tensor
    coverage_unit_index: torch.Tensor
    num_coverage_units: int

    def __post_init__(self) -> None:
        if self.num_schemas < 0:
            raise ValueError("num_schemas must be non-negative.")
        if self.num_coverage_units < 0:
            raise ValueError("num_coverage_units must be non-negative.")
        for name, value in (
            ("schema_ids", self.schema_ids),
            ("coverage_schema_index", self.coverage_schema_index),
            ("coverage_unit_index", self.coverage_unit_index),
        ):
            if value.dtype != torch.long or value.dim() != 1:
                raise TypeError(f"{name} must be a one-dimensional LongTensor.")
        if (
            self.schema_incidence_mass.dim() != 1
            or self.schema_incidence_mass.numel() != self.num_schemas
        ):
            raise ValueError("schema_incidence_mass must have shape [num_schemas].")
        if not self.schema_incidence_mass.is_floating_point():
            raise TypeError("schema_incidence_mass must be a floating-point tensor.")
        if self.coverage_schema_index.numel() != self.coverage_unit_index.numel():
            raise ValueError("Sparse coverage index tensors must have the same length.")
        if self.schema_ids.numel() > 0:
            if self.num_schemas == 0:
                raise ValueError("schema_ids cannot be non-empty when num_schemas is zero.")
            if (
                int(self.schema_ids.min().item()) < 0
                or int(self.schema_ids.max().item()) >= self.num_schemas
            ):
                raise ValueError("schema_ids contains an index outside [0, num_schemas).")
        if self.coverage_schema_index.numel() > 0:
            if self.num_schemas == 0 or self.num_coverage_units == 0:
                raise ValueError("Coverage incidences require positive schema and unit counts.")
            if (
                int(self.coverage_schema_index.min().item()) < 0
                or int(self.coverage_schema_index.max().item()) >= self.num_schemas
            ):
                raise ValueError(
                    "coverage_schema_index contains an index outside [0, num_schemas)."
                )
            if (
                int(self.coverage_unit_index.min().item()) < 0
                or int(self.coverage_unit_index.max().item()) >= self.num_coverage_units
            ):
                raise ValueError(
                    "coverage_unit_index contains an index outside [0, num_coverage_units)."
                )
            keys = self.coverage_unit_index * self.num_schemas + self.coverage_schema_index
            if bool((keys[1:] <= keys[:-1]).any()):
                raise ValueError(
                    "Coverage incidences must be unique and sorted by (unit, schema)."
                )
        if bool((~torch.isfinite(self.schema_incidence_mass)).any()) or bool(
            (self.schema_incidence_mass < 0).any()
        ):
            raise ValueError(
                "schema_incidence_mass must contain finite non-negative values."
            )

    def to(self, device: torch.device | str) -> "DiscoveryMetadata":
        return DiscoveryMetadata(
            num_schemas=self.num_schemas,
            schema_ids=self.schema_ids.to(device),
            schema_incidence_mass=self.schema_incidence_mass.to(device),
            coverage_schema_index=self.coverage_schema_index.to(device),
            coverage_unit_index=self.coverage_unit_index.to(device),
            num_coverage_units=self.num_coverage_units,
        )


def canonical_sparse_coverage(
    coverage_sets: Sequence[Iterable[int]],
    *,
    num_coverage_units: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build unique sparse coverage pairs sorted by unit then schema."""
    if num_coverage_units < 0:
        raise ValueError("num_coverage_units must be non-negative.")
    pairs = sorted(
        {
            (int(unit), schema)
            for schema, units in enumerate(coverage_sets)
            for unit in units
        }
    )
    if not pairs:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()
    if pairs[0][0] < 0 or pairs[-1][0] >= num_coverage_units:
        raise ValueError("Coverage set contains a unit outside [0, num_coverage_units).")
    unit_index = torch.tensor(
        [unit for unit, _ in pairs],
        dtype=torch.long,
        device=device,
    )
    schema_index = torch.tensor(
        [schema for _, schema in pairs],
        dtype=torch.long,
        device=device,
    )
    return schema_index, unit_index


def schema_incidence_costs(
    face_groups: Sequence[object],
    schema_ids: torch.Tensor,
    *,
    num_schemas: int,
    divisor: float = 1.0,
) -> torch.Tensor:
    """Count path plus closure incidences for each materialized schema."""
    if divisor <= 0:
        raise ValueError("divisor must be positive.")
    if schema_ids.dtype != torch.long or schema_ids.dim() != 1:
        raise TypeError("schema_ids must be a one-dimensional LongTensor.")
    if len(face_groups) != int(schema_ids.numel()):
        raise ValueError("schema_ids must contain one entry per face group.")
    costs = torch.zeros(
        num_schemas,
        dtype=torch.float32,
        device=schema_ids.device,
    )
    for group_index, face_group in enumerate(face_groups):
        schema_index = int(schema_ids[group_index].item())
        path_incidences = int(face_group.face_edge_index.numel())
        closure_incidences = int(face_group.closure_edge_index.size(0))
        costs[schema_index] += float(path_incidences + closure_incidences) / divisor
    return costs


def make_discovery_metadata(
    *,
    num_schemas: int,
    face_groups: Sequence[object],
    schema_ids: torch.Tensor,
    coverage_sets: Sequence[Iterable[int]] | None = None,
    num_coverage_units: int = 0,
    schema_incidence_mass: torch.Tensor | None = None,
    cost_divisor: float = 1.0,
) -> DiscoveryMetadata:
    """Construct validated discovery metadata from materialized support."""
    if coverage_sets is None:
        coverage_sets = tuple(() for _ in range(num_schemas))
    if len(coverage_sets) != num_schemas:
        raise ValueError("coverage_sets must contain one set per selected schema.")
    coverage_schema_index, coverage_unit_index = canonical_sparse_coverage(
        coverage_sets,
        num_coverage_units=num_coverage_units,
        device=schema_ids.device,
    )
    if schema_incidence_mass is None:
        schema_incidence_mass = schema_incidence_costs(
            face_groups,
            schema_ids,
            num_schemas=num_schemas,
            divisor=cost_divisor,
        )
    else:
        schema_incidence_mass = schema_incidence_mass.to(
            device=schema_ids.device,
            dtype=torch.float32,
        )
    return DiscoveryMetadata(
        num_schemas=num_schemas,
        schema_ids=schema_ids,
        schema_incidence_mass=schema_incidence_mass,
        coverage_schema_index=coverage_schema_index,
        coverage_unit_index=coverage_unit_index,
        num_coverage_units=num_coverage_units,
    )
