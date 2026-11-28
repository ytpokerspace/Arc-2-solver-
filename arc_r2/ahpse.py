"""Bounded, deterministic ARC hypothesis and program synthesis prototype.

This module proposes short visual transformation programs from the first
visible training pair, then accepts a program only if it exactly reproduces
every supplied training pair. It is deliberately opt-in and does not alter
the default primitive-first benchmark path.

The first milestone covers D4 grid geometry, foreground cropping, and
consistent color remapping. It is a transparent baseline for later typed DSL,
constraint, graph, and adaptive-search integrations.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .grid import Grid, as_grid

# Operations are intentionally small, deterministic, and dependency-free.
_GEOMETRY = (
    "ROT90",
    "ROT180",
    "ROT270",
    "FLIP_H",
    "FLIP_V",
    "TRANSPOSE",
)
_OPERATIONS = ("CROP_FG",) + _GEOMETRY


@dataclass(frozen=True)
class TrainingExample:
    input: Grid
    output: Grid

    @classmethod
    def from_values(cls, input_grid, output_grid) -> "TrainingExample":
        return cls(as_grid(input_grid), as_grid(output_grid))


@dataclass(frozen=True)
class Hypothesis:
    """A declarative candidate; no executable source or dynamic evaluation."""

    operations: tuple[str, ...]
    color_map: tuple[tuple[int, int], ...]

    @property
    def complexity(self) -> int:
        # Color-map entries add a small description-length penalty.
        return len(self.operations) + len(self.color_map)

    def apply(self, grid: Grid) -> Grid:
        result = as_grid(grid)
        for operation in self.operations:
            result = _apply_operation(result, operation)
        mapping = dict(self.color_map)
        return tuple(tuple(mapping.get(value, value) for value in row)
                     for row in result)

    def canonical(self) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
        return self.operations, self.color_map

    def __str__(self) -> str:
        ops = " -> ".join(self.operations) if self.operations else "IDENTITY"
        if self.color_map:
            ops += " | COLORS " + ",".join(f"{a}:{b}" for a, b in self.color_map)
        return ops


@dataclass(frozen=True)
class SynthesisResult:
    hypotheses: tuple[Hypothesis, ...]
    candidates_generated: int
    candidates_checked: int
    exact_fit_count: int
    constraint_checks: int
    constraint_rejections: int
    abstained: bool
    reason: str

    @property
    def best(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None


def _rotate(grid: Grid, turns: int) -> Grid:
    result = grid
    for _ in range(turns % 4):
        height, width = len(result), len(result[0])
        result = tuple(tuple(result[height - 1 - row][col]
                             for row in range(height))
                       for col in range(width))
    return result


def _crop_foreground(grid: Grid) -> Grid:
    points = [(row, col) for row, values in enumerate(grid)
              for col, value in enumerate(values) if value != 0]
    if not points:
        return grid
    top = min(row for row, _ in points)
    bottom = max(row for row, _ in points)
    left = min(col for _, col in points)
    right = max(col for _, col in points)
    return tuple(tuple(grid[row][left:right + 1])
                 for row in range(top, bottom + 1))


def _apply_operation(grid: Grid, operation: str) -> Grid:
    if operation == "CROP_FG":
        return _crop_foreground(grid)
    if operation == "ROT90":
        return _rotate(grid, 1)
    if operation == "ROT180":
        return _rotate(grid, 2)
    if operation == "ROT270":
        return _rotate(grid, 3)
    if operation == "FLIP_H":
        return tuple(tuple(reversed(row)) for row in grid)
    if operation == "FLIP_V":
        return tuple(reversed(grid))
    if operation == "TRANSPOSE":
        return tuple(tuple(grid[row][col] for row in range(len(grid)))
                     for col in range(len(grid[0])))
    raise ValueError(f"unsupported AHPSE operation: {operation}")


def _infer_color_map(source: Grid, target: Grid) -> tuple[tuple[int, int], ...] | None:
    """Infer a deterministic source-color mapping when shapes correspond."""
    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return None
    mapping: dict[int, int] = {}
    for source_row, target_row in zip(source, target):
        for old, new in zip(source_row, target_row):
            previous = mapping.get(old)
            if previous is not None and previous != new:
                return None
            mapping[old] = new
    return tuple(sorted(mapping.items()))


def _programs(max_depth: int, candidate_cap: int) -> Iterable[tuple[str, ...]]:
    yield ()
    emitted = 1
    for depth in range(1, max_depth + 1):
        for operations in product(_OPERATIONS, repeat=depth):
            if emitted >= candidate_cap:
                return
            yield tuple(operations)
            emitted += 1


def synthesize(
    examples: Iterable[TrainingExample],
    *,
    max_depth: int = 2,
    candidate_cap: int = 500,
) -> SynthesisResult:
    """Generate candidates from visible examples and require exact all-pair fit.

    The first example proposes programs. Every supplied example validates them.
    For honest generalization measurement, callers must pass only the fit split
    and score the held-out pair outside this function.
    """
    pairs = tuple(examples)
    if not pairs:
        raise ValueError("at least one training example is required")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be at least 1")

    first = pairs[0]
    accepted: dict[tuple, Hypothesis] = {}
    generated = checked = constraint_checks = constraint_rejections = 0

    for operations in _programs(max_depth, candidate_cap):
        generated += 1
        transformed = first.input
        try:
            for operation in operations:
                transformed = _apply_operation(transformed, operation)
        except (ValueError, IndexError):
            continue
        color_map = _infer_color_map(transformed, first.output)
        if color_map is None:
            continue
        hypothesis = Hypothesis(operations, color_map)
        key = hypothesis.canonical()
        if key in accepted:
            continue
        checked += 1
        # Cheap structural/color constraints run before full pixel equality.
        # The same global color map must explain every visible training pair.
        constraints_ok = True
        for pair in pairs:
            constraint_checks += 1
            candidate_grid = pair.input
            try:
                for operation in operations:
                    candidate_grid = _apply_operation(candidate_grid, operation)
            except (ValueError, IndexError):
                constraints_ok = False
                break
            if ((len(candidate_grid), len(candidate_grid[0])) !=
                    (len(pair.output), len(pair.output[0])) or
                    _infer_color_map(candidate_grid, pair.output) != color_map):
                constraints_ok = False
                break
        if not constraints_ok:
            constraint_rejections += 1
            continue
        if all(_safe_apply(hypothesis, pair.input) == pair.output
               for pair in pairs):
            accepted[key] = hypothesis

    ranked = tuple(sorted(
        accepted.values(),
        key=lambda item: (item.complexity, len(item.operations),
                          item.operations, item.color_map),
    ))
    return SynthesisResult(
        hypotheses=ranked,
        candidates_generated=generated,
        candidates_checked=checked,
        exact_fit_count=len(ranked),
        constraint_checks=constraint_checks,
        constraint_rejections=constraint_rejections,
        abstained=not bool(ranked),
        reason=("exact training fit" if ranked else
                "no candidate exactly fits every supplied training pair"),
    )


def _safe_apply(hypothesis: Hypothesis, grid: Grid) -> Grid | None:
    try:
        return hypothesis.apply(grid)
    except (ValueError, IndexError):
        return None
