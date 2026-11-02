"""Primitive-first ARC solver: explicit grid operations, no MCTS or neural models.

The solver enumerates a small, transparent vocabulary of elementary visual
operations and their two-step compositions. A program is accepted only when
it exactly reproduces every supplied training pair. If no exact fit exists,
the solver abstains instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from collections import deque
from functools import lru_cache

Grid = tuple[tuple[int, ...], ...]


def as_grid(value) -> Grid:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("grid must be a non-empty list of rows")
    width = len(value[0])
    if width < 1:
        raise ValueError("grid rows cannot be empty")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("grid must be rectangular")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 9 for v in row):
            raise ValueError("grid cells must be integers from 0 to 9")
        rows.append(tuple(row))
    return tuple(rows)


def to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def dimensions(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def color_counts(grid: Grid) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in grid:
        for color in row:
            counts[color] = counts.get(color, 0) + 1
    return counts


def background_color(grid: Grid) -> int:
    counts = color_counts(grid)
    return min(counts, key=lambda color: (-counts[color], color))


def foreground_colors(grid: Grid) -> tuple[int, ...]:
    bg = background_color(grid)
    return tuple(sorted(color for color in color_counts(grid) if color != bg))


@dataclass(frozen=True)
class Object:
    color: int
    cells: frozenset[tuple[int, int]]
    bbox: tuple[int, int, int, int]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


def find_objects(grid: Grid, connectivity: int = 4, include_background: bool = False) -> tuple[Object, ...]:
    """Find same-color connected components using 4- or 8-way connectivity."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    bg = background_color(grid)
    h, w = dimensions(grid)
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    seen: set[tuple[int, int]] = set()
    found = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if (r, c) in seen or (not include_background and color == bg):
                continue
            todo = [(r, c)]
            seen.add((r, c))
            cells = set()
            while todo:
                y, x = todo.pop()
                cells.add((y, x))
                for dy, dx in steps:
                    p = (y + dy, x + dx)
                    if 0 <= p[0] < h and 0 <= p[1] < w and p not in seen and grid[p[0]][p[1]] == color:
                        seen.add(p)
                        todo.append(p)
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            found.append(Object(color, frozenset(cells), (min(ys), min(xs), max(ys), max(xs))))
    return tuple(sorted(found, key=lambda obj: (obj.bbox, obj.color)))


def rotate(grid: Grid, turns: int = 1) -> Grid:
    turns %= 4
    result = grid
    for _ in range(turns):
        result = tuple(tuple(result[len(result) - 1 - r][c] for r in range(len(result)))
                       for c in range(len(result[0])))
    return result


def flip_horizontal(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def flip_vertical(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def transpose(grid: Grid) -> Grid:
    return tuple(tuple(grid[r][c] for r in range(len(grid))) for c in range(len(grid[0])))


def reflect_anti_diagonal(grid: Grid) -> Grid:
    h, w = dimensions(grid)
    if h != w:
        # Anti-diagonal reflection is a square-grid operation.
        raise ValueError("anti-diagonal reflection requires a square grid")
    return tuple(tuple(grid[h - 1 - c][w - 1 - r] for c in range(w)) for r in range(h))


def recolor(grid: Grid, source: int, target: int) -> Grid:
    return tuple(tuple(target if cell == source else cell for cell in row) for row in grid)


def swap_colors(grid: Grid, first: int, second: int) -> Grid:
    return tuple(tuple(second if v == first else first if v == second else v for v in row) for row in grid)


def invert_foreground_background(grid: Grid) -> Grid:
    bg = background_color(grid)
    fg = foreground_colors(grid)
    return grid if not fg else swap_colors(grid, bg, fg[0])


def crop_color(grid: Grid, color: int) -> Grid:
    points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color]
    if not points:
        raise ValueError("requested color is absent")
    r0, r1 = min(r for r, _ in points), max(r for r, _ in points)
    c0, c1 = min(c for _, c in points), max(c for _, c in points)
    return tuple(tuple(grid[r][c0:c1 + 1]) for r in range(r0, r1 + 1))


def crop_non_background(grid: Grid) -> Grid:
    colors = foreground_colors(grid)
    if not colors:
        return grid
    points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != background_color(grid)]
    r0, r1 = min(r for r, _ in points), max(r for r, _ in points)
    c0, c1 = min(c for _, c in points), max(c for _, c in points)
    return tuple(tuple(grid[r][c0:c1 + 1]) for r in range(r0, r1 + 1))


def scale(grid: Grid, factor: int) -> Grid:
    if factor < 1:
        raise ValueError("scale factor must be positive")
    return tuple(tuple(cell for cell in row for _ in range(factor))
                 for row in grid for _ in range(factor))


def tile(grid: Grid, rows: int, cols: int) -> Grid:
    if rows < 1 or cols < 1:
        raise ValueError("tile counts must be positive")
    return tuple(tuple(grid[r % len(grid)][c % len(grid[0])] for c in range(len(grid[0]) * cols))
                 for r in range(len(grid) * rows))


def gravity(grid: Grid, direction: str) -> Grid:
    """Compact non-background cells toward one edge, preserving order per lane."""
    if direction not in {"up", "down", "left", "right"}:
        raise ValueError("direction must be up, down, left, or right")
    h, w = dimensions(grid)
    bg = background_color(grid)
    out = [[bg] * w for _ in range(h)]
    if direction in {"up", "down"}:
        for c in range(w):
            values = [grid[r][c] for r in range(h) if grid[r][c] != bg]
            start = 0 if direction == "up" else h - len(values)
            for i, value in enumerate(values):
                out[start + i][c] = value
    else:
        for r in range(h):
            values = [v for v in grid[r] if v != bg]
            start = 0 if direction == "left" else w - len(values)
            for i, value in enumerate(values):
                out[r][start + i] = value
    return tuple(tuple(row) for row in out)


def complete_lines(grid: Grid, axis: str, color: int | None = None) -> Grid:
    """Fill a row/column when it already contains a non-background cell."""
    if axis not in {"rows", "columns"}:
        raise ValueError("axis must be rows or columns")
    bg = background_color(grid)
    fill = color if color is not None else (foreground_colors(grid)[0] if foreground_colors(grid) else bg)
    out = [list(row) for row in grid]
    if axis == "rows":
        for r, row in enumerate(grid):
            if any(v != bg for v in row):
                out[r] = [fill] * len(row)
    else:
        for c in range(len(grid[0])):
            if any(grid[r][c] != bg for r in range(len(grid))):
                for r in range(len(grid)):
                    out[r][c] = fill
    return tuple(tuple(row) for row in out)


def mirror_complete(grid: Grid, axis: str) -> Grid:
    """Copy occupied pixels into their mirror positions across the grid center."""
    if axis not in {"horizontal", "vertical"}:
        raise ValueError("axis must be horizontal or vertical")
    h, w = dimensions(grid)
    bg = background_color(grid)
    out = [list(row) for row in grid]
    for r in range(h):
        for c in range(w):
            rr, cc = (r, w - 1 - c) if axis == "horizontal" else (h - 1 - r, c)
            if grid[r][c] != bg and out[rr][cc] == bg:
                out[rr][cc] = grid[r][c]
    return tuple(tuple(row) for row in out)


def fill_object_boxes(grid: Grid) -> Grid:
    """Fill each detected object's bounding box with that object's color."""
    out = [list(row) for row in grid]
    for obj in find_objects(grid):
        r0, c0, r1, c1 = obj.bbox
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r][c] == background_color(grid):
                    out[r][c] = obj.color
    return tuple(tuple(row) for row in out)


@lru_cache(maxsize=100000)
def apply_primitive(grid: Grid, operation: "Primitive") -> Grid:
    name, args = operation.name, operation.args
    if name == "ROTATE": return rotate(grid, args[0])
    if name == "FLIP_H": return flip_horizontal(grid)
    if name == "FLIP_V": return flip_vertical(grid)
    if name == "TRANSPOSE": return transpose(grid)
    if name == "ANTI_DIAGONAL": return reflect_anti_diagonal(grid)
    if name == "RECOLOR": return recolor(grid, *args)
    if name == "SWAP_COLORS": return swap_colors(grid, *args)
    if name == "INVERT_POLARITY": return invert_foreground_background(grid)
    if name == "CROP_COLOR": return crop_color(grid, args[0])
    if name == "CROP_FOREGROUND": return crop_non_background(grid)
    if name == "SCALE": return scale(grid, args[0])
    if name == "TILE": return tile(grid, *args)
    if name == "GRAVITY": return gravity(grid, args[0])
    if name == "FILL_ROWS": return complete_lines(grid, "rows", args[0])
    if name == "FILL_COLUMNS": return complete_lines(grid, "columns", args[0])
    if name == "MIRROR_H": return mirror_complete(grid, "horizontal")
    if name == "MIRROR_V": return mirror_complete(grid, "vertical")
    if name == "FILL_BOXES": return fill_object_boxes(grid)
    raise ValueError(f"unknown primitive: {name}")


@dataclass(frozen=True, order=True)
class Primitive:
    name: str
    args: tuple = ()

    def __str__(self) -> str:
        return self.name if not self.args else f"{self.name}{self.args}"


@dataclass(frozen=True)
class Program:
    operations: tuple[Primitive, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.operations)

    def apply(self, grid: Grid) -> Grid:
        result = as_grid(grid)
        for operation in self.operations:
            result = apply_primitive(result, operation)
        return result

    def __str__(self) -> str:
        return "IDENTITY" if not self.operations else " -> ".join(map(str, self.operations))


def generate_primitives(training_inputs: tuple[Grid, ...]) -> tuple[Primitive, ...]:
    """Build a finite vocabulary from colors and dimensions visible in fit inputs."""
    if not training_inputs:
        raise ValueError("at least one training input is required")
    grids = tuple(as_grid(g) for g in training_inputs)
    palette = sorted(set().union(*(set(color_counts(g)) for g in grids)))
    max_h = max(len(g) for g in grids)
    max_w = max(len(g[0]) for g in grids)
    ops = {
        Primitive("ROTATE", (turn,)) for turn in (1, 2, 3)
    } | {Primitive(name) for name in ("FLIP_H", "FLIP_V", "TRANSPOSE",
        "ANTI_DIAGONAL", "INVERT_POLARITY", "CROP_FOREGROUND",
        "MIRROR_H", "MIRROR_V", "FILL_BOXES")}
    for a, b in product(palette, repeat=2):
        if a != b:
            ops.add(Primitive("RECOLOR", (a, b)))
            ops.add(Primitive("SWAP_COLORS", (a, b)))
    for color in palette:
        ops.add(Primitive("CROP_COLOR", (color,)))
        ops.add(Primitive("FILL_ROWS", (color,)))
        ops.add(Primitive("FILL_COLUMNS", (color,)))
    for factor in (2, 3):
        ops.add(Primitive("SCALE", (factor,)))
    for rows, cols in ((1, 2), (2, 1), (2, 2), (3, 1), (1, 3)):
        if max_h * rows <= 30 and max_w * cols <= 30:
            ops.add(Primitive("TILE", (rows, cols)))
    for direction in ("up", "down", "left", "right"):
        ops.add(Primitive("GRAVITY", (direction,)))
    return tuple(sorted(ops))


@dataclass(frozen=True)
class FitResult:
    program: Program | None
    candidates_evaluated: int
    exact_fit: bool
    fit_program_count: int


def fit_program(examples: tuple[tuple[Grid, Grid], ...], candidate_cap: int = 5000,
                max_depth: int = 2) -> FitResult:
    """Exhaustively test primitive chains; shortest exact training fit wins.

    No partial-fit program is returned. Enumeration is deterministic and capped.
    """
    if not examples:
        raise ValueError("at least one training pair is required")
    if not 1 <= candidate_cap <= 50000:
        raise ValueError("candidate_cap must be in 1..50000")
    if not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be 0..2")
    pairs = tuple((as_grid(a), as_grid(b)) for a, b in examples)
    primitives = generate_primitives(tuple(a for a, _ in pairs))
    def candidate_programs():
        yield Program()
        for operation in primitives:
            yield Program((operation,))
        if max_depth >= 2:
            for first in primitives:
                for second in primitives:
                    yield Program((first, second))

    best: Program | None = None
    fits = 0
    evaluated = 0
    behavior_seen = set()
    for program in candidate_programs():
        if evaluated >= candidate_cap:
            break
        outputs = []
        valid = True
        for source, target in pairs:
            try:
                predicted = program.apply(source)
            except (ValueError, IndexError, TypeError):
                valid = False
                break
            outputs.append(predicted)
            if predicted != target:
                valid = False
        evaluated += 1
        if not valid:
            continue
        behavior = tuple(outputs)
        if behavior in behavior_seen:
            continue
        behavior_seen.add(behavior)
        fits += 1
        if best is None or (program.complexity, str(program)) < (best.complexity, str(best)):
            best = program
    return FitResult(best, evaluated, best is not None, fits)
