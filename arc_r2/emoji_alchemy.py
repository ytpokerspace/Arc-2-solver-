"""Emoji Alchemy: symbolic ARC transforms and semantic feature comparison.

The emoji tokens are a reversible symbolic alphabet, not decoration. Candidate
programs execute through the canonical grid DSL, then return to emoji space for
object, color, shape, and relation analysis. Integer grids remain authoritative.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
from typing import Iterable

from .dsl import Program
from .emoji_codec import encode_grid, encode_object_graph
from .grid import Grid
from .transforms import OPERATIONS


def apply_emoji_program(grid: Grid, program: Program):
    """Apply a DSL program through the lossless emoji-symbol boundary."""
    from .emoji_codec import decode_grid
    symbolic_input = encode_grid(grid)
    canonical_input = decode_grid(symbolic_input)
    transformed = program.apply(canonical_input)
    return encode_grid(transformed)


def _counter_similarity(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return intersection / union if union else 1.0


@lru_cache(maxsize=50000)
def semantic_signature(grid: Grid) -> dict:
    """Extract stable color, object-shape, and spatial-relation descriptors."""
    symbolic = encode_grid(grid)
    graph = encode_object_graph(grid)
    color_counts = Counter(token for row in symbolic.rows for token in row)
    object_shapes = Counter(
        (obj.color, tuple(sorted(obj.cells)))
        for obj in graph.objects
    )
    object_colors = Counter(obj.color for obj in graph.objects)
    relation_labels = Counter(
        label for _, _, labels in graph.relations for label in labels
    )
    # Coarse normalized location preserves top/bottom and left/right semantics
    # without treating a one-cell translation as a wholly different object.
    locations = Counter()
    for obj in graph.objects:
        r, c = obj.origin
        row_band = min(2, (3 * r) // max(1, graph.height))
        col_band = min(2, (3 * c) // max(1, graph.width))
        locations[(obj.color, row_band, col_band)] += 1
    return {
        "shape": (graph.height, graph.width),
        "color_counts": color_counts,
        "object_shapes": object_shapes,
        "object_colors": object_colors,
        "relations": relation_labels,
        "locations": locations,
    }


def semantic_similarity(predicted: Grid, expected: Grid) -> float:
    """Compare symbolic structure; 1.0 means descriptor-level equivalence."""
    left = semantic_signature(predicted)
    right = semantic_signature(expected)
    color = _counter_similarity(left["color_counts"], right["color_counts"])
    shapes = _counter_similarity(left["object_shapes"], right["object_shapes"])
    objects = _counter_similarity(left["object_colors"], right["object_colors"])
    relations = _counter_similarity(left["relations"], right["relations"])
    locations = _counter_similarity(left["locations"], right["locations"])
    ph, pw = left["shape"]
    eh, ew = right["shape"]
    dimensions = 1.0 if (ph, pw) == (eh, ew) else (
        min(ph, eh) / max(ph, eh) * min(pw, ew) / max(pw, ew)
    )
    return (0.25 * color + 0.25 * shapes + 0.15 * objects
            + 0.15 * relations + 0.10 * locations + 0.10 * dimensions)


def generate_alchemy_candidates(
    existing: Iterable[Program] = (),
    max_depth: int = 2,
) -> tuple[Program, ...]:
    """Generate deterministic symbolic operator hypotheses up to depth two.

    This expands the candidate pool independently of beam pruning. It is
    intentionally bounded; deeper compositions require a separate experiment.
    """
    if not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be between 0 and 2")
    unique = {program.canonical(): program for program in existing}
    identity = Program()
    unique.setdefault(identity.canonical(), identity)
    operations = tuple(OPERATIONS)
    for depth in range(1, max_depth + 1):
        for chain in product(operations, repeat=depth):
            program = Program()
            for operation in chain:
                program = program.then(operation)
            unique.setdefault(program.canonical(), program)
    return tuple(unique[key] for key in sorted(unique))
