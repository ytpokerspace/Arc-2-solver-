"""Immutable grid representation and validation."""
from __future__ import annotations
from typing import Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]

class GridError(ValueError):
    pass

def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    if not isinstance(value, (list, tuple)) or not value:
        raise GridError("grid must be a non-empty list of rows")
    rows = []
    width = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise GridError("each row must be non-empty")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise GridError("grid must be rectangular")
        converted = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise GridError("cells must be integers in the range 0..9")
            converted.append(cell)
        rows.append(tuple(converted))
    return tuple(rows)

def to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]

def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])

def colors(grid: Grid) -> frozenset[int]:
    return frozenset(cell for row in grid for cell in row)

def transpose(grid: Grid) -> Grid:
    return tuple(tuple(grid[r][c] for r in range(len(grid))) for c in range(len(grid[0])))

def count_cells(grid: Grid) -> int:
    return sum(len(row) for row in grid)
