"""Identity-aware object DSL experiment.

Object IDs persist in an active or archived registry. REMOVE hides an object
without destroying its identity; RESTORE is its exact inverse.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, as_grid
from .objects import GridObject, detect_objects
from .object_dsl import ObjectInstruction
from .object_ops import (
    move_object, place_object, reflect_object, rotate_object90, scale_object,
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
    color: int | None = None
    previous_color: int | None = None

    def __post_init__(self):
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        if self.operation in {"REMOVE", "RESTORE"}:
            return
        if self.operation == "RECOLOR":
            for name in ("color", "previous_color"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer for RECOLOR")
                if not 0 <= value <= 9:
                    raise ValueError(f"{name} must be an ARC color in 0..9")
            if self.color == self.previous_color:
                raise ValueError("RECOLOR must change the object's color")
            return
        ObjectInstruction(
            self.operation, 0, factor=self.factor, turns=self.turns,
            axis=self.axis, delta_row=self.delta_row, delta_col=self.delta_col,
            top=self.top, left=self.left,
        )

    def inverse(self) -> "IdentityInstruction":
        if self.operation == "REMOVE":
            return IdentityInstruction(self.object_id, "RESTORE")
        if self.operation == "RESTORE":
            return IdentityInstruction(self.object_id, "REMOVE")
        if self.operation == "RECOLOR":
            return IdentityInstruction(
                self.object_id, "RECOLOR", color=self.previous_color,
                previous_color=self.color,
            )
        inverse = ObjectInstruction(
            self.operation, 0, factor=self.factor, turns=self.turns,
            axis=self.axis, delta_row=self.delta_row, delta_col=self.delta_col,
            top=self.top, left=self.left,
        ).inverse()
        return IdentityInstruction(
            object_id=self.object_id, operation=inverse.operation,
            factor=inverse.factor, turns=inverse.turns, axis=inverse.axis,
            delta_row=inverse.delta_row, delta_col=inverse.delta_col,
            top=inverse.top, left=inverse.left,
        )

    def canonical(self) -> tuple:
        return (
            self.object_id, self.operation, self.factor, self.turns, self.axis,
            self.delta_row, self.delta_col, self.top, self.left,
            self.color, self.previous_color,
        )


@dataclass(frozen=True)
class IdentityState:
    """Grid plus active and archived object registries."""
    height: int
    width: int
    background: int
    objects: tuple[tuple[str, GridObject], ...]
    archived_objects: tuple[tuple[str, GridObject], ...] = ()

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
        return tuple(sorted(
            [object_id for object_id, _ in self.objects]
            + [object_id for object_id, _ in self.archived_objects]
        ))

    def active_object_ids(self) -> tuple[str, ...]:
        return tuple(object_id for object_id, _ in self.objects)


@dataclass(frozen=True)
class IdentityObjectProgram:
    instructions: tuple[IdentityInstruction, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.instructions)

    def then(self, instruction: IdentityInstruction) -> "IdentityObjectProgram":
        if not isinstance(instruction, IdentityInstruction):
            raise TypeError("then requires an IdentityInstruction")
        return IdentityObjectProgram(self.instructions + (instruction,))

    def compose(self, other: "IdentityObjectProgram") -> "IdentityObjectProgram":
        if not isinstance(other, IdentityObjectProgram):
            raise TypeError("compose requires an IdentityObjectProgram")
        return IdentityObjectProgram(self.instructions + other.instructions)

    def canonical(self) -> tuple[tuple, ...]:
        return tuple(instruction.canonical() for instruction in self.instructions)

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
        background = 0 if 0 in counts else min(counts, key=lambda color: (-counts[color], color))
        objects = detect_objects(grid, background=background)
        registry = tuple((f"obj-{index:04d}", obj) for index, obj in enumerate(objects))
        return IdentityState(len(grid), len(grid[0]), background, registry)

    @staticmethod
    def _transform(obj: GridObject, instruction: IdentityInstruction) -> GridObject:
        if instruction.operation == "MOVE":
            return move_object(obj, instruction.delta_row, instruction.delta_col)
        if instruction.operation == "ROTATE90":
            return place_object(rotate_object90(obj, instruction.turns), obj.bbox[0], obj.bbox[1])
        if instruction.operation == "REFLECT":
            return place_object(reflect_object(obj, instruction.axis), obj.bbox[0], obj.bbox[1])
        if instruction.operation == "SCALE":
            return place_object(scale_object(obj, instruction.factor), obj.bbox[0], obj.bbox[1])
        if instruction.operation == "PLACE":
            return place_object(obj, instruction.top, instruction.left)
        if instruction.operation == "RECOLOR":
            if obj.color != instruction.previous_color:
                raise ValueError(
                    f"RECOLOR expected color {instruction.previous_color}, got {obj.color}"
                )
            return GridObject(instruction.color, obj.cells, obj.bbox)
        raise ValueError(f"unsupported transform: {instruction.operation}")

    def apply_state(self, state: IdentityState) -> IdentityState:
        if not isinstance(state, IdentityState):
            raise TypeError("apply_state requires IdentityState; raw grids lose identity metadata")
        active = dict(state.objects)
        archived = dict(state.archived_objects)
        if len(active) != len(state.objects) or len(archived) != len(state.archived_objects):
            raise ValueError("duplicate object identities in state")
        if set(active) & set(archived):
            raise ValueError("an object identity cannot be both active and archived")

        for instruction in self.instructions:
            object_id = instruction.object_id
            if instruction.operation == "REMOVE":
                if object_id not in active:
                    raise ValueError(f"cannot remove inactive object identity: {object_id}")
                archived[object_id] = active.pop(object_id)
                continue
            if instruction.operation == "RESTORE":
                if object_id not in archived:
                    raise ValueError(f"cannot restore unarchived object identity: {object_id}")
                restored = archived[object_id]
                for other_id, other in active.items():
                    if restored.cells & other.cells:
                        raise ValueError(f"restored object {object_id} collides with {other_id}")
                active[object_id] = archived.pop(object_id)
                continue
            if object_id not in active:
                raise ValueError(f"unknown object identity: {object_id}")
            transformed = self._transform(active[object_id], instruction)
            for row, col in transformed.cells:
                if not (0 <= row < state.height and 0 <= col < state.width):
                    raise ValueError("transformed object would leave the grid")
            for other_id, other in active.items():
                if other_id != object_id and transformed.cells & other.cells:
                    raise ValueError("transformed object would collide with another object")
            active[object_id] = transformed

        result = IdentityState(
            state.height, state.width, state.background,
            tuple((object_id, active[object_id]) for object_id in sorted(active)),
            tuple((object_id, archived[object_id]) for object_id in sorted(archived)),
        )
        result.to_grid()
        return result

    def execute(self, grid: Grid) -> IdentityState:
        return self.apply_state(self.initialize(grid))

    def apply(self, grid: Grid) -> Grid:
        return self.execute(grid).to_grid()
