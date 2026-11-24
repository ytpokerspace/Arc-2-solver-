"""Identity-aware object DSL experiment.

Object IDs are assigned once at initialization and retained in IdentityState.
Unlike index-based execution, transformations update the selected object's tracked
geometry directly instead of re-detecting and re-numbering components after each
instruction. Identity metadata is intentionally explicit and is required for
reliable inverse execution after spatial reordering.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, as_grid
from .objects import GridObject, detect_objects
from .object_dsl import ObjectInstruction
from .object_ops import (
    move_object,
    place_object,
    reflect_object,
    rotate_object90,
    scale_object,
)


@dataclass(frozen=True)
class IdentityInstruction:
    object_id: str
    operation: str
    factor: int = 1
    turns: int = 0
    axis: str = "horizontal"
    delta_row: int = 0
    delta_col: int = 0
    top: int = 0
    left: int = 0

    def __post_init__(self):
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        # Reuse the established instruction validator for operation parameters.
        ObjectInstruction(
            self.operation, 0, factor=self.factor, turns=self.turns,
            axis=self.axis, delta_row=self.delta_row, delta_col=self.delta_col,
            top=self.top, left=self.left,
        )

    def inverse(self) -> "IdentityInstruction":
        inverse = ObjectInstruction(
            self.operation, 0, factor=self.factor, turns=self.turns,
            axis=self.axis, delta_row=self.delta_row, delta_col=self.delta_col,
            top=self.top, left=self.left,
        ).inverse()
        return IdentityInstruction(
            object_id=self.object_id,
            operation=inverse.operation,
            factor=inverse.factor,
            turns=inverse.turns,
            axis=inverse.axis,
            delta_row=inverse.delta_row,
            delta_col=inverse.delta_col,
            top=inverse.top,
            left=inverse.left,
        )


@dataclass(frozen=True)
class IdentityState:
    """Grid plus stable object registry; do not discard before inverse execution."""
    height: int
    width: int
    background: int
    objects: tuple[tuple[str, GridObject], ...]

    def to_grid(self) -> Grid:
        output = [[self.background for _ in range(self.width)] for _ in range(self.height)]
        for object_id, obj in self.objects:
            for row, col in obj.cells:
                if not (0 <= row < self.height and 0 <= col < self.width):
                    raise ValueError(f"object {object_id} is outside the grid")
                if output[row][col] != self.background:
                    raise ValueError("tracked objects overlap")
                output[row][col] = obj.color
        return as_grid(output)

    def object_ids(self) -> tuple[str, ...]:
        return tuple(object_id for object_id, _ in self.objects)


@dataclass(frozen=True)
class IdentityObjectProgram:
    instructions: tuple[IdentityInstruction, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.instructions)

    def inverse(self) -> "IdentityObjectProgram":
        return IdentityObjectProgram(tuple(
            instruction.inverse() for instruction in reversed(self.instructions)
        ))

    @staticmethod
    def initialize(grid: Grid) -> IdentityState:
        grid = as_grid(grid)
        counts = {}
        for row in grid:
            for color in row:
                counts[color] = counts.get(color, 0) + 1
        background = min(counts, key=lambda color: (-counts[color], color))
        objects = detect_objects(grid, background=background)
        registry = tuple(
            (f"obj-{index:04d}", obj) for index, obj in enumerate(objects)
        )
        return IdentityState(len(grid), len(grid[0]), background, registry)

    @staticmethod
    def _transform(obj: GridObject, instruction: IdentityInstruction) -> GridObject:
        if instruction.operation == "MOVE":
            return move_object(obj, instruction.delta_row, instruction.delta_col)
        if instruction.operation == "ROTATE90":
            return place_object(
                rotate_object90(obj, instruction.turns), obj.bbox[0], obj.bbox[1]
            )
        if instruction.operation == "REFLECT":
            return place_object(
                reflect_object(obj, instruction.axis), obj.bbox[0], obj.bbox[1]
            )
        if instruction.operation == "SCALE":
            return place_object(
                scale_object(obj, instruction.factor), obj.bbox[0], obj.bbox[1]
            )
        if instruction.operation == "PLACE":
            return place_object(obj, instruction.top, instruction.left)
        raise ValueError(f"unsupported operation: {instruction.operation}")

    def apply_state(self, state: IdentityState) -> IdentityState:
        if not isinstance(state, IdentityState):
            raise TypeError("apply_state requires IdentityState; raw grids lose identity metadata")
        registry = dict(state.objects)
        if len(registry) != len(state.objects):
            raise ValueError("duplicate object identities in state")

        for instruction in self.instructions:
            if instruction.object_id not in registry:
                raise ValueError(f"unknown object identity: {instruction.object_id}")
            selected = registry[instruction.object_id]
            transformed = self._transform(selected, instruction)
            for row, col in transformed.cells:
                if not (0 <= row < state.height and 0 <= col < state.width):
                    raise ValueError("transformed object would leave the grid")
            for other_id, other in registry.items():
                if other_id != instruction.object_id and transformed.cells & other.cells:
                    raise ValueError("transformed object would collide with another object")
            registry[instruction.object_id] = transformed

        result = IdentityState(
            state.height, state.width, state.background,
            tuple((object_id, registry[object_id]) for object_id, _ in state.objects),
        )
        result.to_grid()  # Validate the complete resulting state.
        return result

    def execute(self, grid: Grid) -> IdentityState:
        return self.apply_state(self.initialize(grid))

    def apply(self, grid: Grid) -> Grid:
        """Convenience grid-only API; use execute/apply_state when inversion is needed."""
        return self.execute(grid).to_grid()
