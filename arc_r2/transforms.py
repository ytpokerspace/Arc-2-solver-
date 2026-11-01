"""Explicit grid transformations; reversal, polarity, inverse, and reflection stay distinct."""
from __future__ import annotations
import re
from .grid import Grid, as_grid, transpose

def reverse_horizontal(g: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in g)

def reverse_vertical(g: Grid) -> Grid:
    return tuple(reversed(g))

def rotate_180(g: Grid) -> Grid:
    return reverse_vertical(reverse_horizontal(g))

def reflect_main_diagonal(g: Grid) -> Grid:
    return transpose(g)

def reflect_anti_diagonal(g: Grid) -> Grid:
    return tuple(tuple(g[len(g)-1-c][len(g[0])-1-r]
                       for c in range(len(g))) for r in range(len(g[0])))

def swap_colors(g: Grid, a: int, b: int) -> Grid:
    if a == b:
        return g
    return tuple(tuple(b if x == a else a if x == b else x for x in row) for row in g)

def negate_polarity(g: Grid, foreground: int = 1, background: int = 0) -> Grid:
    """Swap the selected foreground/background labels; not numeric negation."""
    return swap_colors(g, foreground, background)

def inverse_of(name: str) -> str:
    pairs = {
        "REVERSE.HORIZONTAL": "REVERSE.HORIZONTAL",
        "REVERSE.VERTICAL": "REVERSE.VERTICAL",
        "ROTATE.180": "ROTATE.180",
        "REFLECT.MAIN_DIAGONAL": "REFLECT.MAIN_DIAGONAL",
        "REFLECT.ANTI_DIAGONAL": "REFLECT.ANTI_DIAGONAL",
        "COLOR.SWAP.0.1": "COLOR.SWAP.0.1",
        "POLARITY.NEGATE": "POLARITY.NEGATE",
    }
    if name in pairs:
        return pairs[name]
    if re.fullmatch(r"COLOR\\.SWAP\\.[0-9]\\.[0-9]", name):
        return name
    raise ValueError(f"no known inverse for operation: {name}")

def apply_operation(g: Grid, name: str) -> Grid:
    if name == "REVERSE.HORIZONTAL": return reverse_horizontal(g)
    if name == "REVERSE.VERTICAL": return reverse_vertical(g)
    if name == "ROTATE.180": return rotate_180(g)
    if name == "REFLECT.MAIN_DIAGONAL": return reflect_main_diagonal(g)
    if name == "REFLECT.ANTI_DIAGONAL": return reflect_anti_diagonal(g)
    if name == "POLARITY.NEGATE" or name == "COLOR.SWAP.0.1":
        return negate_polarity(g)
    match = re.fullmatch(r"(?:INVERSE\\.)?COLOR\\.SWAP\\.([0-9])\\.([0-9])", name)
    if match:
        return swap_colors(g, int(match.group(1)), int(match.group(2)))
    if name == "PRTE.POLARITY.REVERSE":
        from .prte import reverse_polarity
        return reverse_polarity(g)
    if name == "INVERSE.REVERSE.HORIZONTAL": return reverse_horizontal(g)
    if name == "INVERSE.REVERSE.VERTICAL": return reverse_vertical(g)
    if name == "INVERSE.ROTATE.180": return rotate_180(g)
    if name == "INVERSE.REFLECT.MAIN_DIAGONAL": return reflect_main_diagonal(g)
    if name == "INVERSE.REFLECT.ANTI_DIAGONAL": return reflect_anti_diagonal(g)
    if name == "INVERSE.POLARITY.NEGATE": return negate_polarity(g)
    raise ValueError(f"unknown operation: {name}")

OPERATIONS = (
    "REVERSE.HORIZONTAL", "REVERSE.VERTICAL", "ROTATE.180",
    "REFLECT.MAIN_DIAGONAL", "REFLECT.ANTI_DIAGONAL",
    "POLARITY.NEGATE", "PRTE.POLARITY.REVERSE",
)
