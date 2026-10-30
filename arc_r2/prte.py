"""Polarity-Reversal Transformation Engine (PRTE).

PRTE treats polarity as a relationship between a dominant background label
and a foreground label. It does not perform geometric reversal or reflection.
"""
from __future__ import annotations
from .grid import Grid
from .transforms import swap_colors

def dominant_background(grid: Grid) -> int:
    counts = {}
    for row in grid:
        for color in row: counts[color] = counts.get(color, 0) + 1
    return min(counts, key=lambda color: (-counts[color], color))

def dominant_foreground(grid: Grid, background: int) -> int | None:
    counts = {}
    for row in grid:
        for color in row:
            if color != background: counts[color] = counts.get(color, 0) + 1
    return min(counts, key=lambda color: (-counts[color], color)) if counts else None

def reverse_polarity(grid: Grid) -> Grid:
    """Swap dominant background and dominant non-background color; leave uniform grids unchanged."""
    background = dominant_background(grid)
    foreground = dominant_foreground(grid, background)
    return grid if foreground is None else swap_colors(grid, background, foreground)
