"""Lossless emoji-token representation for ARC color grids."""
from __future__ import annotations
from dataclasses import dataclass
from .grid import Grid, as_grid
from .objects import detect_objects
from .relations import relation_graph

COLOR_TO_EMOJI = {
    0: "⬛", 1: "🟦", 2: "🟥", 3: "🟩", 4: "🟨",
    5: "🟪", 6: "🟧", 7: "🟫", 8: "⬜", 9: "🩶",
}
EMOJI_TO_COLOR = {symbol: color for color, symbol in COLOR_TO_EMOJI.items()}

@dataclass(frozen=True)
class EmojiGrid:
    rows: tuple[tuple[str, ...], ...]
    @property
    def height(self) -> int: return len(self.rows)
    @property
    def width(self) -> int: return len(self.rows[0])

@dataclass(frozen=True)
class EmojiObject:
    color: str
    origin: tuple[int, int]
    cells: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class EmojiObjectGraph:
    height: int
    width: int
    background: str
    objects: tuple[EmojiObject, ...]
    relations: tuple[tuple[int, int, tuple[str, ...]], ...]

def encode_grid(grid: Grid) -> EmojiGrid:
    checked = as_grid(grid)
    return EmojiGrid(tuple(tuple(COLOR_TO_EMOJI[v] for v in row) for row in checked))

def decode_grid(encoded: EmojiGrid | tuple[tuple[str, ...], ...]) -> Grid:
    rows = encoded.rows if isinstance(encoded, EmojiGrid) else encoded
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("emoji grid must contain at least one row")
    width = None
    decoded = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError("emoji grid rows must be non-empty")
        if width is None: width = len(row)
        if len(row) != width: raise ValueError("emoji grid must be rectangular")
        try: decoded.append(tuple(EMOJI_TO_COLOR[token] for token in row))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown emoji cell token: {exc.args[0]!r}") from exc
    return as_grid(decoded)

def encode_object_graph(grid: Grid, background: int | None = None) -> EmojiObjectGraph:
    checked = as_grid(grid)
    if background is None:
        counts = {}
        for row in checked:
            for value in row: counts[value] = counts.get(value, 0) + 1
        background = min(counts, key=lambda color: (-counts[color], color))
    if isinstance(background, bool) or not isinstance(background, int) or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color integer in 0..9")
    objects = detect_objects(checked, background=background)
    encoded = []
    for obj in objects:
        r0, c0, _, _ = obj.bbox
        encoded.append(EmojiObject(COLOR_TO_EMOJI[obj.color], (r0, c0),
            tuple(sorted((r-r0, c-c0) for r, c in obj.cells))))
    relations = tuple((i, j, tuple(sorted(labels)))
        for i, j, labels in relation_graph(objects))
    return EmojiObjectGraph(len(checked), len(checked[0]),
        COLOR_TO_EMOJI[background], tuple(encoded), relations)

def decode_object_graph(graph: EmojiObjectGraph) -> Grid:
    if not isinstance(graph, EmojiObjectGraph):
        raise TypeError("decode_object_graph expects an EmojiObjectGraph")
    if graph.height < 1 or graph.width < 1 or graph.background not in EMOJI_TO_COLOR:
        raise ValueError("invalid object graph dimensions or background")
    rows = [[EMOJI_TO_COLOR[graph.background] for _ in range(graph.width)]
            for _ in range(graph.height)]
    occupied = set()
    for obj in graph.objects:
        if obj.color not in EMOJI_TO_COLOR or not obj.cells:
            raise ValueError("invalid object color or empty object")
        r0, c0 = obj.origin
        for dr, dc in obj.cells:
            r, c = r0+dr, c0+dc
            if not (0 <= r < graph.height and 0 <= c < graph.width):
                raise ValueError("object cell lies outside grid")
            if (r, c) in occupied: raise ValueError("object cells overlap")
            occupied.add((r, c)); rows[r][c] = EMOJI_TO_COLOR[obj.color]
    return as_grid(rows)
