"""Lossless emoji visualization codecs for ARC-R2 grids.

Emoji are a human-readable view only. Integer grids remain the canonical
solver representation; no search operator is changed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, as_grid
from .objects import detect_objects
from .relations import relation_graph

COLOR_TO_EMOJI: dict[int, str] = {
    0: "⬛",
    1: "🟦",
    2: "🟥",
    3: "🟩",
    4: "🟨",
    5: "🟪",
    6: "🟧",
    7: "🟫",
    8: "⬜",
    9: "🩶",
}
EMOJI_TO_COLOR = {symbol: color for color, symbol in COLOR_TO_EMOJI.items()}


@dataclass(frozen=True)
class EmojiGrid:
    """A rectangular grid of emoji tokens; each token represents one ARC cell."""

    rows: tuple[tuple[str, ...], ...]

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])


@dataclass(frozen=True)
class EmojiObject:
    """One monochrome object, stored as a color token and normalized cells."""

    color: str
    origin: tuple[int, int]
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EmojiObjectGraph:
    """Lossless object representation plus spatial-relation metadata."""

    height: int
    width: int
    background: str
    objects: tuple[EmojiObject, ...]
    relations: tuple[tuple[int, int, tuple[str, ...]], ...]


def encode_grid(grid: Grid) -> EmojiGrid:
    """Map each integer cell to exactly one emoji token."""
    checked = as_grid(grid)
    return EmojiGrid(tuple(
        tuple(COLOR_TO_EMOJI[value] for value in row)
        for row in checked
    ))


def decode_grid(encoded: EmojiGrid | tuple[tuple[str, ...], ...]) -> Grid:
    """Restore an emoji grid to its exact integer-grid representation."""
    rows = encoded.rows if isinstance(encoded, EmojiGrid) else encoded
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("emoji grid must contain at least one row")
    decoded = []
    width = None
    for row in rows:
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError("emoji grid rows must be non-empty")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("emoji grid must be rectangular")
        try:
            decoded.append(tuple(EMOJI_TO_COLOR[token] for token in row))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown emoji cell token: {exc.args[0]!r}") from exc
    return as_grid(decoded)


def render_emoji_grid(encoded: EmojiGrid) -> str:
    """Render one emoji token per cell; rows are separated by newlines."""
    if not isinstance(encoded, EmojiGrid):
        raise TypeError("render_emoji_grid expects an EmojiGrid")
    return "\n".join("".join(row) for row in encoded.rows)


def encode_object_graph(grid: Grid, background: int | None = None) -> EmojiObjectGraph:
    """Encode components as emoji-labeled objects while retaining exact cells.

    The default background matches detect_objects(): most frequent color,
    breaking ties by the lowest color value.
    """
    checked = as_grid(grid)
    if background is None:
        counts: dict[int, int] = {}
        for row in checked:
            for value in row:
                counts[value] = counts.get(value, 0) + 1
        background = min(counts, key=lambda color: (-counts[color], color))
    if isinstance(background, bool) or not isinstance(background, int) or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color integer in 0..9")

    objects = detect_objects(checked, background=background)
    encoded_objects = []
    for obj in objects:
        r0, c0, _, _ = obj.bbox
        encoded_objects.append(EmojiObject(
            color=COLOR_TO_EMOJI[obj.color],
            origin=(r0, c0),
            cells=tuple(sorted((r - r0, c - c0) for r, c in obj.cells)),
        ))
    relations = tuple(
        (i, j, tuple(sorted(labels)))
        for i, j, labels in relation_graph(objects)
    )
    return EmojiObjectGraph(
        height=len(checked),
        width=len(checked[0]),
        background=COLOR_TO_EMOJI[background],
        objects=tuple(encoded_objects),
        relations=relations,
    )


def decode_object_graph(graph: EmojiObjectGraph) -> Grid:
    """Reconstruct every cell from the lossless emoji object representation."""
    if not isinstance(graph, EmojiObjectGraph):
        raise TypeError("decode_object_graph expects an EmojiObjectGraph")
    if graph.height < 1 or graph.width < 1:
        raise ValueError("object graph dimensions must be positive")
    if graph.background not in EMOJI_TO_COLOR:
        raise ValueError("unknown background emoji token")

    background = EMOJI_TO_COLOR[graph.background]
    rows = [[background for _ in range(graph.width)] for _ in range(graph.height)]
    occupied: set[tuple[int, int]] = set()
    for obj in graph.objects:
        if obj.color not in EMOJI_TO_COLOR:
            raise ValueError(f"unknown object color token: {obj.color!r}")
        if not obj.cells:
            raise ValueError("objects must contain at least one cell")
        r0, c0 = obj.origin
        color = EMOJI_TO_COLOR[obj.color]
        for dr, dc in obj.cells:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < graph.height and 0 <= c < graph.width):
                raise ValueError("object cell lies outside the grid")
            if (r, c) in occupied:
                raise ValueError("object cells overlap")
            occupied.add((r, c))
            rows[r][c] = color
    return as_grid(rows)
