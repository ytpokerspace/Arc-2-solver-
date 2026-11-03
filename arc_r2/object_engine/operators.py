"""Pure object transformations. Every operation returns a new object."""
from __future__ import annotations

from dataclasses import replace
from .model import ObjectState


def move(obj: ObjectState, dx: int, dy: int) -> ObjectState:
    return replace(obj, x=obj.x + dx, y=obj.y + dy)


def rotate90(obj: ObjectState, anchor: str = "top_left",
             reference_pixel: tuple[int, int] = (0, 0)) -> ObjectState:
    """Rotate clockwise. Center anchoring rejects half-cell translations."""
    h, w = obj.height, obj.width
    pixels = tuple(sorted((c, h - 1 - r, color) for r, c, color in obj.pixels))
    nx, ny = obj.x, obj.y
    if anchor == "center":
        if (w - h) % 2:
            raise ValueError("CENTER_ANCHOR_PARITY: rotation would require half-cell placement")
        nx += (w - h) // 2
        ny += (h - w) // 2
    elif anchor == "reference_pixel":
        rr, rc = reference_pixel
        if not any(r == rr and c == rc for r, c, _ in obj.pixels):
            raise ValueError("REFERENCE_PIXEL_NOT_IN_OBJECT")
        global_r, global_c = obj.y + rr, obj.x + rc
        new_r, new_c = rc, h - 1 - rr
        ny, nx = global_r - new_r, global_c - new_c
    elif anchor != "top_left":
        raise ValueError("UNKNOWN_ANCHOR")
    return replace(obj, pixels=pixels, x=nx, y=ny, rotation=(obj.rotation + 90) % 360)


def flip_horizontal(obj: ObjectState) -> ObjectState:
    return replace(
        obj,
        pixels=tuple(sorted((r, obj.width - 1 - c, color) for r, c, color in obj.pixels)),
        flip_x=not obj.flip_x,
    )


def flip_vertical(obj: ObjectState) -> ObjectState:
    return replace(
        obj,
        pixels=tuple(sorted((obj.height - 1 - r, c, color) for r, c, color in obj.pixels)),
        flip_y=not obj.flip_y,
    )


def scale_up(obj: ObjectState, factor: int) -> ObjectState:
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("SCALE_FACTOR_MUST_BE_POSITIVE_INTEGER")
    if factor == 1:
        return obj
    pixels = []
    for r, c, color in obj.pixels:
        for dr in range(factor):
            for dc in range(factor):
                pixels.append((r * factor + dr, c * factor + dc, color))
    return replace(obj, pixels=tuple(sorted(pixels)), scale=obj.scale * factor)
