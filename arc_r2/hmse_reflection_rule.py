"""A small, explicit reflection rule family for selector extensibility tests."""
from __future__ import annotations

from dataclasses import dataclass

from .primitives import as_grid


@dataclass(frozen=True)
class ReflectionProgram:
    """Reflect a rectangular grid left-right or top-bottom."""

    axis: str

    def apply(self, value):
        grid = as_grid(value)
        if self.axis == "left-right":
            return tuple(tuple(reversed(row)) for row in grid)
        if self.axis == "top-bottom":
            return tuple(reversed(grid))
        raise ValueError("unsupported reflection axis")


def propose_reflection(pairs):
    """Return a unique reflection program only when it exactly fits all pairs."""
    accepted = []
    for axis in ("left-right", "top-bottom"):
        program = ReflectionProgram(axis)
        try:
            if all(program.apply(source) == target for source, target in pairs):
                accepted.append(program)
        except (ValueError, TypeError, IndexError):
            continue
    if len(accepted) == 1:
        return accepted[0], "exact-fit"
    if len(accepted) > 1:
        return None, "reflection-axis-ambiguous"
    return None, "proposal-failed-exact-fit"
