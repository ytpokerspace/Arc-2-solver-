"""Four-connected monochrome object extraction for ARC grids."""
from __future__ import annotations
from dataclasses import dataclass
from .grid import Grid

try:
    import numpy as np
    from scipy.ndimage import label as scipy_label
except ImportError:  # pragma: no cover - portable fallback for minimal installs
    np = None
    scipy_label = None

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
    found = []
    if np is not None and scipy_label is not None:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        array = np.asarray(grid, dtype=np.int8)
        for color in sorted(set(int(value) for value in array.ravel()) - {background}):
            labels, count = scipy_label(array == color, structure=structure)
            for component_id in range(1, count + 1):
                points = np.argwhere(labels == component_id)
                cells = frozenset((int(r), int(c)) for r, c in points)
                rs = points[:, 0]
                cs = points[:, 1]
                found.append(GridObject(
                    color, cells,
                    (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
                ))
        return tuple(sorted(found, key=lambda obj: (obj.bbox, obj.color)))

    seen: set[tuple[int, int]] = set()
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
