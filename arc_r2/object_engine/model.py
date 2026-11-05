"""Immutable object-centric state with local pixels and explicit grid pose."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from ..grid import Grid
from ..objects import detect_objects


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    # Local (row, column, color) pixels; coordinates are relative to bbox origin.
    pixels: tuple[tuple[int, int, int], ...]
    x: int
    y: int
    rotation: int = 0
    flip_x: bool = False
    flip_y: bool = False
    scale: int = 1

    def __post_init__(self) -> None:
        if not self.object_id or not self.pixels:
            raise ValueError("OBJECT_ID_AND_PIXELS_REQUIRED")
        if self.scale < 1 or self.rotation % 90:
            raise ValueError("INVALID_OBJECT_POSE")
        if any(r < 0 or c < 0 or not 0 <= color <= 9 for r, c, color in self.pixels):
            raise ValueError("INVALID_LOCAL_PIXEL")

    @property
    def height(self) -> int:
        return max(r for r, _, _ in self.pixels) + 1

    @property
    def width(self) -> int:
        return max(c for _, c, _ in self.pixels) + 1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.y, self.x, self.y + self.height - 1, self.x + self.width - 1


@dataclass(frozen=True)
class Scene:
    objects: tuple[ObjectState, ...]
    height: int
    width: int
    background: int = 0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("SCENE_DIMENSIONS_MUST_BE_POSITIVE")
        if not 0 <= self.background <= 9:
            raise ValueError("INVALID_BACKGROUND_COLOR")


def extract_scene(grid: Grid, background: int | None = None) -> Scene:
    if not grid or not grid[0]:
        raise ValueError("EMPTY_GRID")
    height, width = len(grid), len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("NON_RECTANGULAR_GRID")
    if background is None:
        counts = Counter(cell for row in grid for cell in row)
        background = min(counts, key=lambda color: (-counts[color], color))
    found = detect_objects(grid, background=background)
    objects = []
    for index, obj in enumerate(found):
        r0, c0, _, _ = obj.bbox
        local = tuple(sorted((r - r0, c - c0, obj.color) for r, c in obj.cells))
        objects.append(ObjectState(f"obj-{index:03d}", local, c0, r0))
    return Scene(tuple(objects), height, width, background)
