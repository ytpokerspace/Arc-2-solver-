"""Deterministic scene renderer with explicit collision and boundary policies."""
from __future__ import annotations

from .model import Scene
from ..grid import Grid


def render_scene(scene: Scene, height: int, width: int,
                 collision: str = "reject") -> Grid:
    if height <= 0 or width <= 0:
        raise ValueError("GRID_DIMENSIONS_MUST_BE_POSITIVE")
    if collision not in {"reject", "first_wins", "last_wins"}:
        raise ValueError("UNKNOWN_COLLISION_POLICY")
    grid = [[scene.background for _ in range(width)] for _ in range(height)]
    occupied: set[tuple[int, int]] = set()
    for obj in scene.objects:
        for r, c, color in obj.pixels:
            gr, gc = obj.y + r, obj.x + c
            if not (0 <= gr < height and 0 <= gc < width):
                raise ValueError("OBJECT_OUT_OF_BOUNDS")
            point = (gr, gc)
            if point in occupied:
                if collision == "reject":
                    raise ValueError("OBJECT_COLLISION")
                if collision == "first_wins":
                    continue
            grid[gr][gc] = color
            occupied.add(point)
    return tuple(tuple(row) for row in grid)
