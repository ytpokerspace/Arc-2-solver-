"""Parameterized, deterministic transforms and descriptors for ARC objects.

These primitives deliberately operate on GridObject values rather than extending
the parameter-free grid DSL. That keeps object geometry independently testable
until the search language can represent operation parameters safely.
"""
from __future__ import annotations

from .objects import GridObject


def _build_object(color: int, cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> GridObject:
    if not cells:
        raise ValueError("an object must contain at least one cell")
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return GridObject(
        color=color,
        cells=frozenset(cells),
        bbox=(min(rows), min(cols), max(rows), max(cols)),
    )


def normalize_object(obj: GridObject) -> GridObject:
    """Move the object's top-left bounding-box corner to (0, 0)."""
    top, left, _, _ = obj.bbox
    return _build_object(obj.color, {(r - top, c - left) for r, c in obj.cells})


def object_size(obj: GridObject) -> tuple[int, int, int]:
    """Return (cell_count, bounding_box_height, bounding_box_width)."""
    return obj.size, obj.height, obj.width


def shape_signature(obj: GridObject) -> tuple[tuple[int, int], ...]:
    """Return a translation-invariant, sorted cell pattern."""
    return tuple(sorted(normalize_object(obj).cells))


def move_object(obj: GridObject, delta_row: int, delta_col: int) -> GridObject:
    """Translate an object by a relative (row, column) offset."""
    if isinstance(delta_row, bool) or not isinstance(delta_row, int):
        raise TypeError("delta_row must be an integer")
    if isinstance(delta_col, bool) or not isinstance(delta_col, int):
        raise TypeError("delta_col must be an integer")
    return _build_object(
        obj.color,
        {(r + delta_row, c + delta_col) for r, c in obj.cells},
    )


def place_object(obj: GridObject, top: int, left: int) -> GridObject:
    """Place the object's bounding-box top-left corner at an absolute location."""
    if isinstance(top, bool) or not isinstance(top, int):
        raise TypeError("top must be an integer")
    if isinstance(left, bool) or not isinstance(left, int):
        raise TypeError("left must be an integer")
    normalized = normalize_object(obj)
    return move_object(normalized, top, left)


def scale_object(obj: GridObject, factor: int) -> GridObject:
    """Scale each cell into a factor-by-factor block, anchored at the top-left."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    normalized = normalize_object(obj)
    cells = {
        (r * factor + dr, c * factor + dc)
        for r, c in normalized.cells
        for dr in range(factor)
        for dc in range(factor)
    }
    return _build_object(obj.color, cells)


def rotate_object90(obj: GridObject, turns: int = 1) -> GridObject:
    """Rotate clockwise in 90-degree increments, normalizing after rotation."""
    if isinstance(turns, bool) or not isinstance(turns, int):
        raise TypeError("turns must be an integer")
    result = normalize_object(obj)
    for _ in range(turns % 4):
        height = result.height
        result = _build_object(
            obj.color,
            {(c, height - 1 - r) for r, c in result.cells},
        )
    return normalize_object(result)


def reflect_object(obj: GridObject, axis: str = "horizontal") -> GridObject:
    """Reflect across the horizontal or vertical axis of the bounding box.

    'horizontal' mirrors left-to-right; 'vertical' mirrors top-to-bottom.
    """
    normalized = normalize_object(obj)
    if axis == "horizontal":
        cells = {(r, normalized.width - 1 - c) for r, c in normalized.cells}
    elif axis == "vertical":
        cells = {(normalized.height - 1 - r, c) for r, c in normalized.cells}
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'")
    return normalize_object(_build_object(obj.color, cells))
