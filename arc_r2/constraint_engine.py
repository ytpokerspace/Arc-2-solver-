"""Cheap, exact necessary-condition checks for ARC candidate programs.

Checks are ordered from inexpensive metadata to full cell equality. They never
accept a candidate on a heuristic: a program is valid only after exact equality
on every training pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable

Grid = tuple[tuple[int, ...], ...]\n\ndef dimensions(grid: Grid) -> tuple[int, int]:\n    return len(grid), len(grid[0])


@dataclass(frozen=True)
class ConstraintCheck:
    accepted: bool
    reason: str
    exact_comparison_performed: bool


def check_prediction(predicted: Grid, expected: Grid) -> ConstraintCheck:
    """Reject mismatched shape/palette/histogram before the final exact check."""
    if dimensions(predicted) != dimensions(expected):
        return ConstraintCheck(False, "shape", False)

    predicted_counts = Counter(value for row in predicted for value in row)
    expected_counts = Counter(value for row in expected for value in row)
    if predicted_counts.keys() != expected_counts.keys():
        return ConstraintCheck(False, "palette", False)
    if predicted_counts != expected_counts:
        return ConstraintCheck(False, "color_counts", False)
    if predicted != expected:
        return ConstraintCheck(False, "cell_positions", True)
    return ConstraintCheck(True, "exact_match", True)


def check_program_on_examples(program, examples: Iterable[tuple[Grid, Grid]]) -> tuple[bool, int, dict[str, int]]:
    """Check every example, stopping at the first violated constraint."""
    counters = {"shape": 0, "palette": 0, "color_counts": 0,
                "cell_positions": 0, "exact_match": 0, "execution_error": 0}
    for source, expected in examples:
        try:
            predicted = program.apply(source)
        except (ValueError, IndexError, TypeError):
            counters["execution_error"] += 1
            return False, 0, counters
        result = check_prediction(predicted, expected)
        counters[result.reason] += 1
        if not result.accepted:
            return False, int(result.exact_comparison_performed), counters
    return True, sum(counters[key] for key in ("cell_positions", "exact_match")), counters
