"""Four-connected monochrome object extraction for ARC grids."""
from __future__ import annotations
from dataclasses import dataclass
from .grid import Grid

@dataclass(frozen=True)
class GridObject:
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

def detect_objects(grid: Grid, background: int | None = None) -> tuple[GridObject, ...]:
    """Extract 4-connected same-color components; background defaults to most frequent color."""
    if background is None:
        counts = {}
        for row in grid:
            for value in row: counts[value] = counts.get(value, 0) + 1
        background = min(counts, key=lambda color: (-counts[color], color))
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    found = []
    for r in range(height):
        for c in range(width):
            color = grid[r][c]
            if color == background or (r, c) in seen: continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = set()
            while stack:
                cr, cc = stack.pop()
                cells.add((cr, cc))
                for nr, nc in ((cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr,nc) not in seen and grid[nr][nc] == color:
                        seen.add((nr,nc)); stack.append((nr,nc))
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            found.append(GridObject(color, frozenset(cells), (min(rs), min(cs), max(rs), max(cs))))
    return tuple(sorted(found, key=lambda obj: (obj.bbox, obj.color)))
