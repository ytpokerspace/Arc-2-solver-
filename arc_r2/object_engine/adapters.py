"""Grid-level adapters for parameterized object transformations."""
from __future__ import annotations

from collections.abc import Callable

from ..grid import Grid
from .model import Scene, extract_scene
from .operators import move, rotate90, flip_horizontal, flip_vertical, scale_up
from .renderer import render_scene

ObjectTransform = Callable[..., object]


def transform_object(grid: Grid, index: int, transform: Callable,
                     *args, background: int | None = None,
                     collision: str = "reject", **kwargs) -> Grid:
    scene = extract_scene(grid, background=background)
    if not 0 <= index < len(scene.objects):
        raise ValueError("OBJECT_INDEX_OUT_OF_RANGE")
    objects = list(scene.objects)
    objects[index] = transform(objects[index], *args, **kwargs)
    return render_scene(
        Scene(tuple(objects), scene.height, scene.width, scene.background),
        collision=collision,
    )


def move_object(grid: Grid, index: int, dx: int, dy: int,
                background: int | None = None) -> Grid:
    return transform_object(grid, index, move, dx, dy, background=background)


def rotate_object(grid: Grid, index: int, anchor: str = "top_left",
                  reference_pixel: tuple[int, int] = (0, 0),
                  background: int | None = None) -> Grid:
    return transform_object(
        grid, index, rotate90, anchor=anchor,
        reference_pixel=reference_pixel, background=background,
    )


def flip_object_horizontal(grid: Grid, index: int,
                           background: int | None = None) -> Grid:
    return transform_object(grid, index, flip_horizontal, background=background)


def flip_object_vertical(grid: Grid, index: int,
                         background: int | None = None) -> Grid:
    return transform_object(grid, index, flip_vertical, background=background)


def scale_object_up(grid: Grid, index: int, factor: int,
                    background: int | None = None) -> Grid:
    return transform_object(grid, index, scale_up, factor, background=background)


def object_exists(index: int, background: int | None = None) -> Callable[[Grid], bool]:
    def check(grid: Grid) -> bool:
        return 0 <= index < len(extract_scene(grid, background=background).objects)
    return check
