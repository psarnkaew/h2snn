"""Schemas with direction (D7, plan §9.1): sequences of (relation, direction) over the
relations kept by the fact layer.

A hop is `(relation, direction)` with `relation` a compact id after the fact layer and
`direction` +1 (traverse the fact along its stored orientation), -1 (against it) or 0 (a
self-inverse relation, whose two directions are the same facts). A schema and its reverse
describe the same cells, so they share one canonical key (`schema_key`) and one face group.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.config import MetaPathConfig
from src.data.fact_layer import FactLayer

Hop = tuple[int, int]


def reverse_hops(hops: tuple[Hop, ...]) -> tuple[Hop, ...]:
    """Hops of the same walk read from the other end."""
    return tuple((relation, -direction) for relation, direction in reversed(hops))


def schema_key(node_types: tuple[str, ...], hops: tuple[Hop, ...]) -> tuple:
    """Canonical identity of a schema up to reversal (the smaller of the two readings)."""
    forward = (tuple(node_types), tuple(hops))
    backward = (tuple(reversed(node_types)), reverse_hops(tuple(hops)))
    return min(forward, backward)


@dataclass(frozen=True, slots=True)
class ResolvedSchema:
    name: str
    node_types: tuple[str, ...]
    hops: tuple[Hop, ...]
    sample_per_anchor: int | None

    @property
    def key(self) -> tuple:
        return schema_key(self.node_types, self.hops)

    @property
    def length(self) -> int:
        return len(self.hops)


def _hop_types(relation_tuple: tuple[str, str, str], direction: int) -> tuple[str, str]:
    src, _, dst = relation_tuple
    return (dst, src) if direction < 0 else (src, dst)


def _render(layer: FactLayer, relation: int, direction: int) -> str:
    name = layer.kept_names[relation]
    return f"({name}, {'+' if direction > 0 else '-' if direction < 0 else 'self-inverse'})"


def resolve_schema(
    template: MetaPathConfig,
    layer: FactLayer,
    relation_tuples: tuple[tuple[str, str, str], ...],
) -> ResolvedSchema:
    """Resolve a configured schema to `(relation, direction)` hops under `layer`.

    `relation_tuples` are the typed tuples of the kept relations (compact order). With
    explicit `template.hops`, each named relation resolves through `layer.resolve` (a dropped
    inverse becomes its kept partner with the direction flipped). Without hops, every
    node-type transition must have exactly one candidate (relation, direction); otherwise
    the error names the transition and the candidates. Under `untied` only stored
    directions (+) are candidates and explicit '-' hops are rejected.
    """
    types = tuple(template.node_types)
    if len(types) < 2:
        raise ValueError(f"schema {template.name!r} needs at least two node types.")
    self_inverse = set(layer.self_inverse)
    hops: list[Hop] = []
    if template.hops is not None:
        if len(template.hops) != len(types) - 1:
            raise ValueError(
                f"schema {template.name!r}: {len(template.hops)} hops for "
                f"{len(types)} node types."
            )
        for j, (name, sign) in enumerate(template.hops):
            if layer.mode == "untied" and sign == "-":
                raise ValueError(
                    f"schema {template.name!r}: explicit '-' hop {name!r} under "
                    "fact_layer: untied (untied walks stored directions only)."
                )
            relation, base = layer.resolve(name)
            direction = 0 if relation in self_inverse else base * (1 if sign == "+" else -1)
            if _hop_types(relation_tuples[relation], direction) != (types[j], types[j + 1]):
                raise ValueError(
                    f"schema {template.name!r}: hop {j + 1} {name!r} {sign} does not join "
                    f"{types[j]!r} -> {types[j + 1]!r}."
                )
            hops.append((relation, direction))
    else:
        for j in range(len(types) - 1):
            transition = (types[j], types[j + 1])
            candidates: list[Hop] = []
            for relation, relation_tuple in enumerate(relation_tuples):
                if relation in self_inverse:
                    if _hop_types(relation_tuple, 1) == transition:
                        candidates.append((relation, 0))
                    continue
                directions = (1,) if layer.mode == "untied" else (1, -1)
                for direction in directions:
                    if _hop_types(relation_tuple, direction) == transition:
                        candidates.append((relation, direction))
            if len(candidates) != 1:
                rendered = ", ".join(_render(layer, r, s) for r, s in candidates) or "none"
                raise ValueError(
                    f"schema {template.name!r}: transition {transition[0]!r} -> "
                    f"{transition[1]!r} resolves to {len(candidates)} (relation, direction) "
                    f"candidates [{rendered}]; spell it out with 'hops'."
                )
            hops.append(candidates[0])
    return ResolvedSchema(template.name, types, tuple(hops), template.sample_per_anchor)


def resolve_schemas(
    templates: tuple[MetaPathConfig, ...],
    layer: FactLayer,
    relation_tuples: tuple[tuple[str, str, str], ...],
) -> tuple[ResolvedSchema, ...]:
    """Resolve every template; a schema whose reverse is already present joins it (one
    group per canonical key; the first template's name and quota are kept)."""
    resolved: list[ResolvedSchema] = []
    seen: set[tuple] = set()
    for template in templates:
        schema = resolve_schema(template, layer, relation_tuples)
        if schema.key in seen:
            continue
        seen.add(schema.key)
        resolved.append(schema)
    return tuple(resolved)
