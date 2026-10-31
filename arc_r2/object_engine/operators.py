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


def normalize(obj: ObjectState) -> ObjectState:
    """Normalize local pixels while preserving the object's global top-left."""
    min_r = min(r for r, _, _ in obj.pixels)
    min_c = min(c for _, c, _ in obj.pixels)
    pixels = tuple(sorted((r - min_r, c - min_c, color)
                          for r, c, color in obj.pixels))
    return replace(obj, pixels=pixels, x=obj.x + min_c, y=obj.y + min_r)


def rotate180(obj: ObjectState) -> ObjectState:
    return rotate90(rotate90(obj))


def rotate270(obj: ObjectState) -> ObjectState:
    return rotate90(rotate180(obj))


def transpose(obj: ObjectState) -> ObjectState:
    return normalize(replace(
        obj, pixels=tuple((c, r, color) for r, c, color in obj.pixels)
    ))


def scale_down(obj: ObjectState, factor: int) -> ObjectState:
    """Project local cells into factor-sized blocks; this can be lossy."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("SCALE_FACTOR_MUST_BE_POSITIVE_INTEGER")
    if factor == 1:
        return obj
    pixels = tuple(sorted({(r // factor, c // factor, color)
                           for r, c, color in obj.pixels}))
    return replace(obj, pixels=pixels, scale=max(1, obj.scale // factor))


def hollow(obj: ObjectState) -> ObjectState:
    """Keep boundary pixels; remove pixels with all four orthogonal neighbors."""
    occupied = {(r, c) for r, c, _ in obj.pixels}
    pixels = tuple(
        (r, c, color) for r, c, color in obj.pixels
        if not all((r + dr, c + dc) in occupied
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    )
    return replace(obj, pixels=pixels)


def recolor(obj: ObjectState, color: int) -> ObjectState:
    if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 9:
        raise ValueError("INVALID_COLOR")
    return replace(obj, pixels=tuple((r, c, color) for r, c, _ in obj.pixels))


def bbox(obj: ObjectState) -> tuple[int, int, int, int]:
    """Return local (min_row, min_col, max_row, max_col), inclusive."""
    rows = [r for r, _, _ in obj.pixels]
    cols = [c for _, c, _ in obj.pixels]
    return min(rows), min(cols), max(rows), max(cols)


def place_at(obj: ObjectState, x: int, y: int) -> ObjectState:
    return replace(obj, x=x, y=y)


def center_in(obj: ObjectState, height: int, width: int) -> ObjectState:
    if height <= 0 or width <= 0:
        raise ValueError("CANVAS_DIMENSIONS_MUST_BE_POSITIVE")
    r0, c0, r1, c1 = bbox(obj)
    shape_h, shape_w = r1 - r0 + 1, c1 - c0 + 1
    # Center the actual bounding box, including non-normalized local pixels.
    return place_at(obj, (width - shape_w) // 2 - c0,
                    (height - shape_h) // 2 - r0)


def align_to(obj: ObjectState, target: ObjectState, edge: str = "left") -> ObjectState:
    """Align global bounding-box edges; supports left, right, top, and bottom."""
    or0, oc0, or1, oc1 = bbox(obj)
    tr0, tc0, tr1, tc1 = bbox(target)
    if edge == "left":
        return place_at(obj, target.x + tc0 - oc0, obj.y)
    if edge == "right":
        return place_at(obj, target.x + tc1 - oc1, obj.y)
    if edge == "top":
        return place_at(obj, obj.x, target.y + tr0 - or0)
    if edge == "bottom":
        return place_at(obj, obj.x, target.y + tr1 - or1)
    raise ValueError("UNKNOWN_ALIGNMENT_EDGE")
