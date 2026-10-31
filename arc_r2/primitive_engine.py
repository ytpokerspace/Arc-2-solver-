"""Typed primitive registry. Catalog data and executable implementations stay separate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .grid import Grid

GridTransform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    family: str
    input_type: str = "grid"
    output_type: str = "grid"
    cost: float = 1.0
    tags: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class Primitive:
    spec: PrimitiveSpec
    apply: GridTransform


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Primitive] = {}

    def register(self, spec: PrimitiveSpec, apply: GridTransform) -> None:
        if not spec.name or not spec.family:
            raise ValueError("PRIMITIVE_NAME_AND_FAMILY_REQUIRED")
        if spec.name in self._items:
            raise ValueError(f"DUPLICATE_PRIMITIVE:{spec.name}")
        if spec.input_type != "grid" or spec.output_type != "grid":
            raise ValueError("CURRENT_SEARCH_ADAPTER_REQUIRES_GRID_TO_GRID_PRIMITIVES")
        if spec.cost <= 0:
            raise ValueError("PRIMITIVE_COST_MUST_BE_POSITIVE")
        self._items[spec.name] = Primitive(spec, apply)

    def get(self, name: str) -> Primitive:
        return self._items[name]

    def enabled(self) -> tuple[Primitive, ...]:
        return tuple(p for p in self._items.values() if p.spec.enabled)

    def validate_catalog(self, expected_count: int = 200) -> None:
        actual = len(self._items)
        if actual != expected_count:
            raise ValueError(f"PRIMITIVE_CATALOG_COUNT_MISMATCH:expected={expected_count}:actual={actual}")

    def __len__(self) -> int:
        return len(self._items)
