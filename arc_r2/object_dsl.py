"""Parameterized object-level DSL with bounded instruction generation.

Object indices refer to the deterministic ordering returned by detect_objects
at the moment each instruction is applied. The grid canvas and background are
preserved; transforms that leave the canvas or collide with another object fail
closed with ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, as_grid
from .objects import detect_objects
from .object_ops import (
    move_object, normalize_object, object_size, place_object,
    reflect_object, rotate_object90, scale_object, shape_signature,
)


@dataclass(frozen=True)
class ObjectInstruction:
    operation: str
    object_index: int
    factor: int = 1
    turns: int = 0
    axis: str = "horizontal"
    delta_row: int = 0
    delta_col: int = 0
    top: int = 0
    left: int = 0

    def __post_init__(self):
        if self.operation not in {"SCALE", "ROTATE90", "REFLECT", "MOVE", "PLACE"}:
            raise ValueError(f"unsupported object operation: {self.operation}")
        if isinstance(self.object_index, bool) or not isinstance(self.object_index, int) or self.object_index < 0:
            raise ValueError("object_index must be a non-negative integer")
        for name in ("factor", "turns", "delta_row", "delta_col", "top", "left"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.operation == "SCALE" and self.factor < 1:
            raise ValueError("factor must be positive")
        if self.operation == "ROTATE90" and self.turns % 4 == 0:
            raise ValueError("rotation must change orientation")
        if self.operation == "REFLECT" and self.axis not in {"horizontal", "vertical"}:
            raise ValueError("axis must be horizontal or vertical")

    def canonical(self) -> tuple:
        return (self.operation, self.object_index, self.factor, self.turns,
                self.axis, self.delta_row, self.delta_col, self.top, self.left)

    def __str__(self) -> str:
        if self.operation == "SCALE":
            args = f"object={self.object_index}, factor={self.factor}"
        elif self.operation == "ROTATE90":
            args = f"object={self.object_index}, turns={self.turns}"
        elif self.operation == "REFLECT":
            args = f"object={self.object_index}, axis={self.axis}"
        elif self.operation == "MOVE":
            args = f"object={self.object_index}, dr={self.delta_row}, dc={self.delta_col}"
        else:
            args = f"object={self.object_index}, top={self.top}, left={self.left}"
        return f"{self.operation}({args})"


@dataclass(frozen=True)
class ObjectProgram:
    instructions: tuple[ObjectInstruction, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.instructions)

    def canonical(self) -> tuple[tuple, ...]:
        return tuple(item.canonical() for item in self.instructions)

    def then(self, instruction: ObjectInstruction) -> "ObjectProgram":
        return ObjectProgram(self.instructions + (instruction,))

    def apply(self, grid: Grid) -> Grid:
        result = as_grid(grid)
        for instruction in self.instructions:
            result = apply_object_instruction(result, instruction)
        return result

    def __str__(self) -> str:
        return "IDENTITY" if not self.instructions else " -> ".join(map(str, self.instructions))


def object_descriptors(grid: Grid) -> tuple[dict, ...]:
    """Expose measurable size/shape descriptors for every detected object."""
    return tuple({
        "index": index,
        "color": obj.color,
        "size": object_size(obj),
        "shape": shape_signature(obj),
        "bbox": obj.bbox,
    } for index, obj in enumerate(detect_objects(grid)))


def apply_object_instruction(grid: Grid, instruction: ObjectInstruction) -> Grid:
    grid = as_grid(grid)
    objects = detect_objects(grid)
    if instruction.object_index >= len(objects):
        raise ValueError("object_index is outside the detected object list")
    selected = objects[instruction.object_index]
    if instruction.operation == "SCALE":
        transformed = scale_object(selected, instruction.factor)
    elif instruction.operation == "ROTATE90":
        transformed = rotate_object90(selected, instruction.turns)
    elif instruction.operation == "REFLECT":
        transformed = reflect_object(selected, instruction.axis)
    elif instruction.operation == "MOVE":
        transformed = move_object(selected, instruction.delta_row, instruction.delta_col)
    else:
        transformed = place_object(selected, instruction.top, instruction.left)

    height, width = len(grid), len(grid[0])
    if any(r < 0 or c < 0 or r >= height or c >= width for r, c in transformed.cells):
        raise ValueError("transformed object would leave the grid")
    occupied_by_others = set().union(*(obj.cells for i, obj in enumerate(objects) if i != instruction.object_index)) if len(objects) > 1 else set()
    if transformed.cells & occupied_by_others:
        raise ValueError("transformed object would collide with another object")

    counts = {}
    for row in grid:
        for color in row:
            counts[color] = counts.get(color, 0) + 1
    background = min(counts, key=lambda color: (-counts[color], color))
    output = [list(row) for row in grid]
    for r, c in selected.cells:
        output[r][c] = background
    for r, c in transformed.cells:
        output[r][c] = transformed.color
    return as_grid(output)


def generate_object_instructions(
    training_inputs: tuple[Grid, ...],
    parameter_cap: int = 1200,
    max_offset: int = 2,
) -> tuple[ObjectInstruction, ...]:
    """Generate a deterministic, finite candidate vocabulary from task dimensions."""
    if not training_inputs:
        raise ValueError("at least one training input is required")
    if parameter_cap < 1 or parameter_cap > 10000:
        raise ValueError("parameter_cap must be in 1..10000")
    if max_offset < 1 or max_offset > 5:
        raise ValueError("max_offset must be in 1..5")
    grids = tuple(as_grid(grid) for grid in training_inputs)
    max_objects = max(len(detect_objects(grid)) for grid in grids)
    if max_objects == 0:
        return ()
    max_height = max(len(grid) for grid in grids)
    max_width = max(len(grid[0]) for grid in grids)
    candidates: list[ObjectInstruction] = []
    seen = set()

    def add(item):
        key = item.canonical()
        if key not in seen and len(candidates) < parameter_cap:
            seen.add(key)
            candidates.append(item)

    anchors = sorted({
        (0, 0), (0, max_width - 1), (max_height - 1, 0),
        (max_height - 1, max_width - 1),
        ((max_height - 1) // 2, (max_width - 1) // 2),
    })
    for index in range(max_objects):
        for turns in (1, 2, 3):
            add(ObjectInstruction("ROTATE90", index, turns=turns))
        for axis in ("horizontal", "vertical"):
            add(ObjectInstruction("REFLECT", index, axis=axis))
        for factor in (2, 3):
            add(ObjectInstruction("SCALE", index, factor=factor))
        for dr in range(-max_offset, max_offset + 1):
            for dc in range(-max_offset, max_offset + 1):
                if dr or dc:
                    add(ObjectInstruction("MOVE", index, delta_row=dr, delta_col=dc))
        for top, left in anchors:
            add(ObjectInstruction("PLACE", index, top=top, left=left))
        if len(candidates) >= parameter_cap:
            break
    return tuple(candidates)
